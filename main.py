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
    return {"recipes": recipes}

@app.post("/api/load")
def load_recipe_data(req: LoadRequest):
    recipe_name = req.recipe_name
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

# ================== 核心求解与灵敏度分析 ==================
def run_lp(ing_data, req_data):
    # 1. 提取基础数据
    ingredients = [i['原料名称'] for i in ing_data]
    n_ing = len(ingredients)
    
    df_ing = pd.DataFrame(ing_data)
    df_req = pd.DataFrame(req_data)
    
    nutrient_names = df_req['营养素'].tolist()
    n_nut = len(nutrient_names)

    lb_nut_base = df_req['下限'].values
    ub_nut_base = df_req['上限'].values
    content = df_ing[nutrient_names].values
    price_base = df_ing['价格'].values
    lb_ing_base = df_ing['用量下限%'].values / 100
    ub_ing_base = df_ing['用量上限%'].values / 100

    # 2. 构建 Pyomo 模型
    model = pyo.ConcreteModel()
    
    model.x = pyo.Var(ingredients)
    model.price = pyo.Param(ingredients, initialize=lambda m, i: price_base[ingredients.index(i)], mutable=True)
    model.lb_ing = pyo.Param(ingredients, initialize=lambda m, i: lb_ing_base[ingredients.index(i)], mutable=True)
    model.ub_ing = pyo.Param(ingredients, initialize=lambda m, i: ub_ing_base[ingredients.index(i)], mutable=True)
    
    model.lb_nut = pyo.Param(nutrient_names, initialize=lambda m, j: lb_nut_base[nutrient_names.index(j)], mutable=True)
    model.ub_nut = pyo.Param(nutrient_names, initialize=lambda m, j: ub_nut_base[nutrient_names.index(j)], mutable=True)

    model.c_ing_lb = pyo.Constraint(ingredients, rule=lambda m, i: m.x[i] >= m.lb_ing[i])
    model.c_ing_ub = pyo.Constraint(ingredients, rule=lambda m, i: m.x[i] <= m.ub_ing[i])
    
    def nut_rule_lb(m, j):
        return sum(content[ingredients.index(i), nutrient_names.index(j)] * m.x[i] for i in ingredients) >= m.lb_nut[j]
    def nut_rule_ub(m, j):
        return sum(content[ingredients.index(i), nutrient_names.index(j)] * m.x[i] for i in ingredients) <= m.ub_nut[j]
    model.c_nut_lb = pyo.Constraint(nutrient_names, rule=nut_rule_lb)
    model.c_nut_ub = pyo.Constraint(nutrient_names, rule=nut_rule_ub)
    
    model.c_total = pyo.Constraint(expr=sum(model.x[i] for i in ingredients) == 1)
    model.cost = pyo.Objective(expr=sum(model.price[i] * model.x[i] for i in ingredients), sense=pyo.minimize)

    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)

    solver = SolverFactory('highs')
    try:
        res = solver.solve(model, tee=False, load_solutions=True)
        if res.solver.status != pyo.SolverStatus.ok or res.solver.termination_condition != pyo.TerminationCondition.optimal:
            return None
    except Exception:
        return None

    # 3. 提取基准解
    cost_base = pyo.value(model.cost)
    x_base = {i: pyo.value(model.x[i]) for i in ingredients}
    rc_base = {i: model.rc.get(model.x[i], 0.0) for i in ingredients}
    
    dual_ing_lb = {i: model.dual.get(model.c_ing_lb[i], 0.0) for i in ingredients}
    dual_ing_ub = {i: model.dual.get(model.c_ing_ub[i], 0.0) for i in ingredients}
    dual_nut_lb = {j: model.dual.get(model.c_nut_lb[j], 0.0) for j in nutrient_names}
    dual_nut_ub = {j: model.dual.get(model.c_nut_ub[j], 0.0) for j in nutrient_names}

    # 4. 二分搜索核心逻辑
    TOL_X = 1e-4
    
    def solve_attempt():
        try:
            res = solver.solve(model, tee=False, load_solutions=True)
            return res.solver.status == pyo.SolverStatus.ok and res.solver.termination_condition == pyo.TerminationCondition.optimal
        except Exception:
            return False

    def basis_unchanged():
        return all(abs(pyo.value(model.x[i]) - x_base[i]) <= TOL_X for i in ingredients)

    def binary_search_bound(param, base_val, physical_limit, direction):
        step = max(abs(base_val) * 0.1, 0.1)
        low, high = base_val, base_val
        
        if direction == 'increase':
            while True:
                high = base_val + step
                if high > physical_limit:
                    high = physical_limit
                    param.set_value(high)
                    if not solve_attempt() or not basis_unchanged(): break
                    param.set_value(base_val); solve_attempt(); return high - base_val
                param.set_value(high)
                if not solve_attempt() or not basis_unchanged(): break
                low = high; step *= 2
            while high - low > max(0.01, abs(high)*1e-4):
                mid = (low + high) / 2
                param.set_value(mid)
                if solve_attempt() and basis_unchanged(): low = mid
                else: high = mid
            param.set_value(base_val); solve_attempt(); return low - base_val
        else:
            while True:
                high = base_val - step
                if high < physical_limit:
                    high = physical_limit
                    param.set_value(high)
                    if not solve_attempt() or not basis_unchanged(): break
                    param.set_value(base_val); solve_attempt(); return base_val - high
                param.set_value(high)
                if not solve_attempt() or not basis_unchanged(): break
                low = high; step *= 2
            while low - high > max(0.01, abs(low)*1e-4):
                mid = (low + high) / 2
                param.set_value(mid)
                if solve_attempt() and basis_unchanged(): low = mid
                else: high = mid
            param.set_value(base_val); solve_attempt(); return base_val - low

    # 5. 计算原料灵敏度
    result_recipe = []
    for i, ing in enumerate(ingredients):
        qty = x_base[ing] * 100
        
        # 【微调】：限制价格搜索范围为当前价格的 10% ~ 10 倍
        min_p_limit = max(0.0, price_base[i] * 0.1)
        max_p_limit = price_base[i] * 10.0
        
        p_inc = binary_search_bound(model.price[ing], price_base[i], max_p_limit, 'increase')
        p_dec = binary_search_bound(model.price[ing], price_base[i], min_p_limit, 'decrease')
        
        ing_inc_lb = binary_search_bound(model.lb_ing[ing], lb_ing_base[i], 1.0, 'increase')
        ing_dec_lb = binary_search_bound(model.lb_ing[ing], lb_ing_base[i], 0.0, 'decrease')
        ing_inc_ub = binary_search_bound(model.ub_ing[ing], ub_ing_base[i], 1.0, 'increase')
        ing_dec_ub = binary_search_bound(model.ub_ing[ing], ub_ing_base[i], 0.0, 'decrease')

        result_recipe.append({
            "原料名称": ing,
            "结果%": round(qty, 4),
            "递减成本": round(rc_base[ing], 4),
            "当前价格": float(price_base[i]),
            "价格下限": round(float(price_base[i] - p_dec), 2),
            "价格上限": round(float(price_base[i] + p_inc), 2),
            "下限影子价格": round(float(dual_ing_lb[ing]), 4),
            "上限影子价格": round(float(dual_ing_ub[ing]), 4),
            "下限允许增加": round(float(ing_inc_lb * 100), 2),
            "下限允许减少": round(float(ing_dec_lb * 100), 2),
            "上限允许增加": round(float(ing_inc_ub * 100), 2),
            "上限允许减少": round(float(ing_dec_ub * 100), 2)
        })

    # 6. 计算营养素灵敏度
    result_nutrients = []
    for j, nut in enumerate(nutrient_names):
        actual = sum(content[ingredients.index(i), nutrient_names.index(nut)] * x_base[i] for i in ingredients)
        
        n_inc_lb = binary_search_bound(model.lb_nut[nut], lb_nut_base[j], 10000.0, 'increase') 
        n_dec_lb = binary_search_bound(model.lb_nut[nut], lb_nut_base[j], 0.0, 'decrease')
        n_inc_ub = binary_search_bound(model.ub_nut[nut], ub_nut_base[j], 10000.0, 'increase')
        n_dec_ub = binary_search_bound(model.ub_nut[nut], ub_nut_base[j], 0.0, 'decrease')
        
        result_nutrients.append({
            "营养素": nut,
            "结果": round(float(actual), 4),
            "下限": float(lb_nut_base[j]),
            "上限": float(ub_nut_base[j]),
            "状态": "达标" if (lb_nut_base[j] - 1e-4) <= actual <= (ub_nut_base[j] + 1e-4) else "不达标",
            "下界影子价格": round(float(dual_nut_lb[nut]), 4),
            "下界允许增加": round(float(n_inc_lb), 2),
            "下界允许减少": round(float(n_dec_lb), 2),
            "上界影子价格": round(float(dual_nut_ub[nut]), 4),
            "上界允许增加": round(float(n_inc_ub), 2),
            "上界允许减少": round(float(n_dec_ub), 2)
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

# ================== Excel 导出 ==================
def beautify_excel(writer, sheet_name, df, start_row=0):
    worksheet = writer.sheets[sheet_name]
    for col_idx, col in enumerate(df.columns, start=1):
        max_len = max(df[col].astype(str).map(len).max(), len(str(col))) + 2
        col_letter = chr(64 + col_idx) if col_idx <= 26 else "A" + chr(64 + col_idx - 26)
        worksheet.column_dimensions[col_letter].width = max(max_len, 10)
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
        df_summary = pd.DataFrame({'项目': ['最优总成本'], '数值': [result['total_cost']], '单位': ['元/吨']})
        df_recipe = pd.DataFrame(result['recipe'])
        df_recipe_export = df_recipe[['原料名称', '结果%', '当前价格', '价格下限', '价格上限']]
        df_summary.to_excel(writer, sheet_name="配方与成本", index=False, startrow=0)
        df_recipe_export.to_excel(writer, sheet_name="配方与成本", index=False, startrow=3)
        beautify_excel(writer, "配方与成本", df_summary, 0)
        beautify_excel(writer, "配方与成本", df_recipe_export, 3)

        df_nut = pd.DataFrame(result['nutrients'])
        df_nut_export = df_nut[['营养素', '结果', '下限', '上限', '状态', '下界影子价格', '上界影子价格']]
        df_nut_export.to_excel(writer, sheet_name="营养分析", index=False)
        beautify_excel(writer, "营养分析", df_nut_export)

        df_sens_nut = df_nut[['营养素', '下界允许增加', '下界允许减少', '上界允许增加', '上界允许减少']]
        df_sens_nut.to_excel(writer, sheet_name="营养灵敏度", index=False)
        beautify_excel(writer, "营养灵敏度", df_sens_nut)

        df_sens_ing = df_recipe[['原料名称', '下限影子价格', '下限允许增加', '下限允许减少', '上限影子价格', '上限允许增加', '上限允许减少']]
        df_sens_ing.to_excel(writer, sheet_name="原料灵敏度", index=False)
        beautify_excel(writer, "原料灵敏度", df_sens_ing)

    excel_buffer.seek(0)
    return Response(
        content=excel_buffer.getvalue(), 
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
        headers={"Content-Disposition": "attachment; filename=feed_formula_report.xlsx"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)