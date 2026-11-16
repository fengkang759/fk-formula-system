from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core import Suffix
import os
import glob
import io
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SolveRequest(BaseModel):
    ingredients: list
    requirements: list

class LoadRequest(BaseModel):
    recipe_name: str

@app.get("/")
def read_index():
    return FileResponse("index.html")

@app.get("/api/files")
def get_available_files():
    files = glob.glob('*_ingredients.csv')
    recipes = [f.replace('_ingredients.csv', '') for f in files]
    print(f"后端扫描到的配方: {recipes}")
    return {"recipes": recipes}

# ================== 1. 本地加载接口 ==================
@app.post("/api/load")
def load_recipe_data(req: LoadRequest):
    recipe_name = req.recipe_name
    print(f"正在尝试加载配方: {recipe_name}")
    
    ing_file = f"{recipe_name}_ingredients.csv"
    nut_file = f"{recipe_name}_nutrition.csv"
    
    if not os.path.exists(ing_file) or not os.path.exists(nut_file):
        return {"status": "error", "message": f"找不到文件"}
    
    try:
        df_ing = pd.read_csv(ing_file, encoding='utf-8-sig')
        df_nut = pd.read_csv(nut_file, encoding='utf-8-sig')
        
        df_ing.columns = df_ing.columns.str.strip()
        df_nut.columns = df_nut.columns.str.strip()
        df_nut = df_nut.fillna(0)
        
        return {
            "status": "success",
            "ingredients": df_ing.to_dict(orient='records'),
            "requirements": df_nut.to_dict(orient='records')
        }
    except Exception as e:
        return {"status": "error", "message": f"文件解析失败: {str(e)}"}

# ================== 2. 自定义上传接口 ==================
@app.post("/api/upload")
async def upload_csv(ing_file: UploadFile = File(...), nut_file: UploadFile = File(...)):
    try:
        content_ing = await ing_file.read()
        content_nut = await nut_file.read()
        
        df_ing = pd.read_csv(io.BytesIO(content_ing), encoding='utf-8-sig')
        df_nut = pd.read_csv(io.BytesIO(content_nut), encoding='utf-8-sig')
        
        df_ing.columns = df_ing.columns.str.strip()
        df_nut.columns = df_nut.columns.str.strip()
        df_nut = df_nut.fillna(0)
        
        return {
            "status": "success",
            "ingredients": df_ing.to_dict(orient='records'),
            "requirements": df_nut.to_dict(orient='records')
        }
    except Exception as e:
        return {"status": "error", "message": f"文件解析失败: {str(e)}"}

# ================== 3. 核心求解逻辑 ==================
def run_lp(ing_data, req_data):
    ingredients = [i['原料名称'] for i in ing_data]
    n_ing = len(ingredients)
    
    df_ing = pd.DataFrame(ing_data)
    df_req = pd.DataFrame(req_data)
    
    nutrient_names = df_req['营养素'].tolist()
    lb_nut_base = df_req['下限'].values
    ub_nut_base = df_req['上限'].values
    content = df_ing[nutrient_names].values

    price_base = df_ing['价格'].values
    lb_ing = df_ing['用量下限%'].values / 100
    ub_ing = df_ing['用量上限%'].values / 100

    model = pyo.ConcreteModel()
    model.x = pyo.Var(ingredients, bounds=lambda m, i: (lb_ing[ingredients.index(i)], ub_ing[ingredients.index(i)]))
    model.price = pyo.Param(range(n_ing), initialize=price_base.tolist(), mutable=True, within=pyo.Reals)
    model.lb_nut = pyo.Param(range(len(nutrient_names)), initialize=lb_nut_base.tolist(), mutable=True, within=pyo.Reals)
    model.ub_nut = pyo.Param(range(len(nutrient_names)), initialize=ub_nut_base.tolist(), mutable=True, within=pyo.Reals)
    model.cost = pyo.Objective(expr=sum(model.price[i] * model.x[ingredients[i]] for i in range(n_ing)), sense=pyo.minimize)
    
    model.nutrient_constraints = pyo.ConstraintList()
    for j, nut in enumerate(nutrient_names):
        expr = sum(content[i, j] * model.x[ingredients[i]] for i in range(n_ing))
        model.nutrient_constraints.add(expr >= model.lb_nut[j])
        model.nutrient_constraints.add(expr <= model.ub_nut[j])
    model.total_constraint = pyo.Constraint(expr=sum(model.x[i] for i in ingredients) == 1)

    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)

    solver = SolverFactory('highs')
    try:
        res = solver.solve(model, tee=False, load_solutions=True)
        if res.solver.status != pyo.SolverStatus.ok or res.solver.termination_condition != pyo.TerminationCondition.optimal:
            return None
    except Exception:
        return None

    cost_base = pyo.value(model.cost)
    x_base = {i: pyo.value(model.x[ingredients[i]]) for i in range(n_ing)}
    dual_total_base = model.dual.get(model.total_constraint, None)
    dual_lb_base = {j: model.dual.get(model.nutrient_constraints[j*2 + 1], None) for j in range(len(nutrient_names))}
    dual_ub_base = {j: model.dual.get(model.nutrient_constraints[j*2 + 2], None) for j in range(len(nutrient_names))}
    rc_base = {i: model.rc.get(model.x[ingredients[i]], None) for i in range(n_ing)}

    TOL = 0.01 
    PRICE_MIN, PRICE_MAX = 0.0, 1e7
    NUT_MIN, NUT_MAX = 0.0, 100.0
    energy_keywords = ['代谢能', '能量KC', '虾代谢能KC', '禽代谢能KC', '代谢能KC']
    ENERGY_IDX = next((nutrient_names.index(k) for k in energy_keywords if k in nutrient_names), None)
    NUT_MAX_ENERGY = 10000

    def get_nut_phys_bound(j, is_lb):
        return (NUT_MIN if is_lb else NUT_MAX_ENERGY) if (ENERGY_IDX is not None and j == ENERGY_IDX) else (NUT_MIN if is_lb else NUT_MAX)

    def solve_attempt():
        try:
            res = solver.solve(model, tee=False, load_solutions=True)
            if res.solver.status == pyo.SolverStatus.ok and res.solver.termination_condition == pyo.TerminationCondition.optimal:
                return True
        except Exception:
            pass
        return False

    def basis_unchanged(mode='price'):
        if mode == 'price':
            return all(abs(pyo.value(model.x[ingredients[i]]) - x_base[i]) <= TOL for i in range(n_ing))
        else:
            dual_total_new = model.dual.get(model.total_constraint, None)
            if abs((dual_total_new or 0) - (dual_total_base or 0)) > TOL: return False
            for j in range(len(nutrient_names)):
                lb_new = model.dual.get(model.nutrient_constraints[j*2 + 1], None)
                ub_new = model.dual.get(model.nutrient_constraints[j*2 + 2], None)
                if abs((lb_new or 0) - (dual_lb_base[j] or 0)) > TOL: return False
                if abs((ub_new or 0) - (dual_ub_base[j] or 0)) > TOL: return False
            return True

    def price_range(i, direction):
        base = price_base[i]
        step = max(base * 0.1, 1.0)
        if direction == 'increase':
            low, high = base, base
            while True:
                high = base + step
                if high > PRICE_MAX:
                    high = PRICE_MAX; model.price[i] = high
                    if not solve_attempt() or not basis_unchanged('price'): break
                    else: model.price[i] = base; solve_attempt(); return high - base
                model.price[i] = high
                if not solve_attempt() or not basis_unchanged('price'): break
                low = high; step *= 2
            while high - low > TOL:
                mid = (low + high) / 2; model.price[i] = mid
                if solve_attempt() and basis_unchanged('price'): low = mid
                else: high = mid
            model.price[i] = base; solve_attempt(); return low - base
        else: 
            low, high = base, base
            while True:
                high = base - step
                if high < PRICE_MIN:
                    high = PRICE_MIN; model.price[i] = high
                    if not solve_attempt() or not basis_unchanged('price'): break
                    else: model.price[i] = base; solve_attempt(); return base - high
                model.price[i] = high
                if not solve_attempt() or not basis_unchanged('price'): break
                low = high; step *= 2
            while low - high > TOL:
                mid = (low + high) / 2; model.price[i] = mid
                if solve_attempt() and basis_unchanged('price'): low = mid
                else: high = mid
            model.price[i] = base; solve_attempt(); return base - low

    def nutrient_range(j, bound_type, direction):
        param = model.lb_nut if bound_type == 'lb' else model.ub_nut
        base = lb_nut_base[j] if bound_type == 'lb' else ub_nut_base[j]
        phys_low, phys_high = get_nut_phys_bound(j, True), get_nut_phys_bound(j, False)
        step = max(base * 0.1, 0.1)
        if direction == 'increase':
            low, high = base, base
            while True:
                high = base + step
                if high > phys_high:
                    high = phys_high; param[j] = high
                    if not solve_attempt() or not basis_unchanged('rhs'): break
                    else: param[j] = base; solve_attempt(); return high - base
                param[j] = high
                if not solve_attempt() or not basis_unchanged('rhs'): break
                low = high; step *= 2
            while high - low > TOL:
                mid = (low + high) / 2; param[j] = mid
                if solve_attempt() and basis_unchanged('rhs'): low = mid
                else: high = mid
            param[j] = base; solve_attempt(); return low - base
        else:
            low, high = base, base
            while True:
                high = base - step
                if high < phys_low:
                    high = phys_low; param[j] = high
                    if not solve_attempt() or not basis_unchanged('rhs'): break
                    else: param[j] = base; solve_attempt(); return base - high
                param[j] = high
                if not solve_attempt() or not basis_unchanged('rhs'): break
                low = high; step *= 2
            while low - high > TOL:
                mid = (low + high) / 2; param[j] = mid
                if solve_attempt() and basis_unchanged('rhs'): low = mid
                else: high = mid
            param[j] = base; solve_attempt(); return base - low

    result_recipe = []
    for i, ing in enumerate(ingredients):
        qty = pyo.value(model.x[ing]) * 100
        inc = price_range(i, 'increase')
        dec = price_range(i, 'decrease')
        result_recipe.append({
            "原料名称": str(ing),
            "结果%": round(float(qty), 4),
            "递减成本": round(float(rc_base[i] or 0.0), 4),
            "当前价格": float(price_base[i]),
            "价格下限": round(float(price_base[i] - dec), 2),
            "价格上限": round(float(price_base[i] + inc), 2)
        })

    result_nutrients = []
    for j, nut in enumerate(nutrient_names):
        actual = sum(content[i, j] * pyo.value(model.x[ingredients[i]]) for i in range(n_ing))
        
        inc_lb = nutrient_range(j, 'lb', 'increase') if pd.notna(lb_nut_base[j]) else 0
        dec_lb = nutrient_range(j, 'lb', 'decrease') if pd.notna(lb_nut_base[j]) else 0
        inc_ub = nutrient_range(j, 'ub', 'increase') if pd.notna(ub_nut_base[j]) else 0
        dec_ub = nutrient_range(j, 'ub', 'decrease') if pd.notna(ub_nut_base[j]) else 0
        
        result_nutrients.append({
            "营养素": str(nut),
            "结果": round(float(actual), 4),
            "下限": float(lb_nut_base[j]) if pd.notna(lb_nut_base[j]) else 0.0,
            "上限": float(ub_nut_base[j]) if pd.notna(ub_nut_base[j]) else 999.0,
            "状态": "达标" if (lb_nut_base[j] - 0.001) <= actual <= (ub_nut_base[j] + 0.001) else "不达标",
            "下界影子价格": round(float(dual_lb_base[j] or 0.0), 4),
            "下界允许增加": round(float(inc_lb), 2),
            "下界允许减少": round(float(dec_lb), 2),
            "上界影子价格": round(float(dual_ub_base[j] or 0.0), 4),
            "上界允许增加": round(float(inc_ub), 2),
            "上界允许减少": round(float(dec_ub), 2)
        })

    return {
        "status": "success",
        "total_cost": round(float(cost_base), 2),
        "recipe": result_recipe,
        "nutrients": result_nutrients
    }

@app.post("/api/solve")
def solve_formula(req: SolveRequest):
    try:
        result = run_lp(req.ingredients, req.requirements)
        if result is None:
            return {"status": "error", "message": "无可行解！请检查营养指标或原料限制。"}
        return result
    except Exception as e:
        return {"status": "error", "message": f"计算出错: {str(e)}"}

# ================== 4. 导出 Excel ==================
def beautify_excel(writer, sheet_name, df, start_row=0):
    worksheet = writer.sheets[sheet_name]
    for col_idx, col in enumerate(df.columns, start=1):
        max_len = max(df[col].astype(str).map(len).max(), len(str(col))) + 2
        col_letter = chr(64 + col_idx) if col_idx <= 26 else "A" + chr(64 + col_idx - 26)
        worksheet.column_dimensions[col_letter].width = max_len
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
        center_align = Alignment(horizontal="center", vertical="center")
        thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))
        header_cell = worksheet.cell(row=start_row + 1, column=col_idx)
        header_cell.font, header_cell.fill, header_cell.alignment, header_cell.border = header_font, header_fill, center_align, thin_border
        for row_idx in range(start_row + 2, start_row + 2 + len(df)):
            cell = worksheet.cell(row=row_idx, column=col_idx)
            cell.alignment, cell.border = center_align, thin_border

@app.post("/api/export")
def export_excel(req: SolveRequest):
    result = run_lp(req.ingredients, req.requirements)
    if result is None:
        return Response(content="无可行解，无法导出", media_type="text/plain")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        sheet_name = "配方与成本"
        df_summary = pd.DataFrame({'项目': ['最优总成本'], '数值': [result['total_cost']], '单位': ['元/吨']})
        df_recipe = pd.DataFrame(result['recipe'])
        df_summary.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
        df_recipe.to_excel(writer, sheet_name=sheet_name, index=False, startrow=3)
        beautify_excel(writer, sheet_name, df_summary, 0)
        beautify_excel(writer, sheet_name, df_recipe, 3)

        df_nut = pd.DataFrame(result['nutrients'])
        df_nut_export = df_nut[['营养素', '结果', '下限', '上限', '状态', '下界影子价格', '上界影子价格']]
        df_nut_export.to_excel(writer, sheet_name="营养分析", index=False)
        beautify_excel(writer, "营养分析", df_nut_export)

        df_sens = df_nut[['营养素', '下界允许增加', '下界允许减少', '上界允许增加', '上界允许减少']]
        df_sens.to_excel(writer, sheet_name="灵敏度边界", index=False)
        beautify_excel(writer, "灵敏度边界", df_sens)

    excel_buffer.seek(0)
    return Response(
        content=excel_buffer.getvalue(), 
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
        headers={"Content-Disposition": "attachment; filename=feed_formula_report.xlsx"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)