from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core import Suffix
import os
import glob
import io
import re
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_BOUND = 1e5
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
VALID_RECIPE_RE = re.compile(r'^[\w\u4e00-\u9fa5\-]+$')


class SolveRequest(BaseModel):
    ingredients: list
    requirements: list


class LoadRequest(BaseModel):
    recipe_name: str


# ================== 工具函数 ==================
def parse_numeric(value, default=0.0):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        s = value.strip()
        if s == '' or s.lower() in ('∞', 'inf', 'infinity', 'none', '-', 'nan'):
            return default
        try:
            return float(s)
        except ValueError:
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_with_fallback(content_bytes):
    last_err = None
    for enc in ('utf-8-sig', 'gbk', 'gb18030', 'utf-8'):
        try:
            df = pd.read_csv(io.BytesIO(content_bytes), encoding=enc)
            df.columns = df.columns.astype(str).str.strip()
            return df
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last_err = e
            continue
    raise ValueError(f"无法解析 CSV 文件。错误：{last_err}")


def get_solver():
    # 必须使用传统 highs，因为 appsi_highs 不支持可变参数在二分查找中的反复更新
    for name in ('highs', 'glpk'):
        try:
            s = SolverFactory(name)
            if s.available(exception_flag=False):
                return s, name
        except Exception:
            continue
    return None, None


def _beautify_header(writer, sheet_name, df, start_row=0):
    ws = writer.sheets[sheet_name]
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
    center_align = Alignment(horizontal="center", vertical="center")
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    for col_idx in range(1, len(df.columns) + 1):
        header_cell = ws.cell(row=start_row + 1, column=col_idx)
        header_cell.font = header_font
        header_cell.fill = header_fill
        header_cell.alignment = center_align
        header_cell.border = thin_border
        for row_idx in range(start_row + 2, start_row + 2 + len(df)):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.alignment = center_align
            cell.border = thin_border


def _autosize_columns(writer, sheet_name, dfs, max_cols=20):
    ws = writer.sheets[sheet_name]
    for col_idx in range(1, max_cols + 1):
        col_letter = get_column_letter(col_idx)
        max_len = 10
        for df in dfs:
            if col_idx <= len(df.columns):
                col_name = df.columns[col_idx - 1]
                try:
                    values_len = df[col_name].astype(str).map(len).max()
                    if not pd.isna(values_len):
                        max_len = max(max_len, int(values_len), len(str(col_name)))
                except Exception:
                    pass
        ws.column_dimensions[col_letter].width = min(max_len + 2, 50)


# ================== 路由 ==================
@app.get("/")
def read_index():
    return FileResponse("index.html")


@app.get("/api/files")
def get_available_files():
    files = glob.glob('*_ingredients.csv')
    recipes = sorted(f.replace('_ingredients.csv', '') for f in files)
    return {"recipes": recipes}


@app.post("/api/load")
def load_recipe_data(req: LoadRequest):
    recipe_name = req.recipe_name or ""
    if not VALID_RECIPE_RE.match(recipe_name):
        return {"status": "error", "message": "非法的配方名称"}

    available = [f.replace('_ingredients.csv', '') for f in glob.glob('*_ingredients.csv')]
    if recipe_name not in available:
        return {"status": "error", "message": f"找不到配方 '{recipe_name}'"}

    ing_file = f"{recipe_name}_ingredients.csv"
    nut_file = f"{recipe_name}_nutrition.csv"
    if not os.path.exists(nut_file):
        return {"status": "error", "message": f"找不到营养文件 '{nut_file}'"}

    try:
        with open(ing_file, 'rb') as f:
            df_ing = read_csv_with_fallback(f.read())
        with open(nut_file, 'rb') as f:
            df_nut = read_csv_with_fallback(f.read())
        return {
            "status": "success",
            "ingredients": df_ing.where(pd.notnull(df_ing), None).to_dict(orient='records'),
            "requirements": df_nut.where(pd.notnull(df_nut), None).to_dict(orient='records'),
        }
    except Exception as e:
        return {"status": "error", "message": f"文件解析失败: {str(e)}"}


@app.post("/api/upload")
async def upload_csv(ing_file: UploadFile = File(...), nut_file: UploadFile = File(...)):
    try:
        content_ing = await ing_file.read()
        content_nut = await nut_file.read()
        if len(content_ing) > MAX_UPLOAD_SIZE or len(content_nut) > MAX_UPLOAD_SIZE:
            return {"status": "error", "message": "文件过大（最大 10MB）"}

        df_ing = read_csv_with_fallback(content_ing)
        df_nut = read_csv_with_fallback(content_nut)
        return {
            "status": "success",
            "ingredients": df_ing.where(pd.notnull(df_ing), None).to_dict(orient='records'),
            "requirements": df_nut.where(pd.notnull(df_nut), None).to_dict(orient='records'),
        }
    except Exception as e:
        return {"status": "error", "message": f"文件解析失败: {str(e)}"}


# ================== 校验 ==================
def validate_inputs(df_ing, df_req, nutrient_names):
    errors = []
    if df_ing.empty:
        errors.append("原料数据为空")
    if df_req.empty:
        errors.append("营养需求数据为空")
    if errors:
        return errors

    for col in ('原料名称', '价格', '用量下限%', '用量上限%'):
        if col not in df_ing.columns:
            errors.append(f"原料文件缺少列: {col}")
    for col in ('营养素', '下限', '上限'):
        if col not in df_req.columns:
            errors.append(f"营养文件缺少列: {col}")
    if errors:
        return errors

    names = df_ing['原料名称'].astype(str).tolist()
    if len(names) != len(set(names)):
        dups = sorted({n for n in names if names.count(n) > 1})
        errors.append(f"原料名称重复: {dups}")

    nuts = df_req['营养素'].astype(str).tolist()
    if len(nuts) != len(set(nuts)):
        dups = sorted({n for n in nuts if nuts.count(n) > 1})
        errors.append(f"营养素名称重复: {dups}")

    for i, row in df_ing.iterrows():
        name = row.get('原料名称', f'第{i + 1}行')
        lb = parse_numeric(row.get('用量下限%'), 0)
        ub = parse_numeric(row.get('用量上限%'), 100)
        price = parse_numeric(row.get('价格'), 0)
        if lb < 0:
            errors.append(f"原料 '{name}' 用量下限为负: {lb}")
        if ub > 100:
            errors.append(f"原料 '{name}' 用量上限超过 100: {ub}")
        if lb > ub + 1e-9:
            errors.append(f"原料 '{name}' 用量下限({lb})大于上限({ub})")
        if price < 0:
            errors.append(f"原料 '{name}' 价格为负: {price}")

    for i, row in df_req.iterrows():
        name = row.get('营养素', f'第{i + 1}行')
        lb = parse_numeric(row.get('下限'), 0)
        ub = parse_numeric(row.get('上限'), 1e9)
        if lb > ub + 1e-9:
            errors.append(f"营养素 '{name}' 下限({lb})大于上限({ub})")

    missing_cols = [n for n in nutrient_names if n not in df_ing.columns]
    if missing_cols:
        errors.append(f"原料文件缺少营养素列: {missing_cols}")

    return errors


# ================== 核心求解与灵敏度分析 ==================
def run_lp(ing_data, req_data):
    try:
        df_ing = pd.DataFrame(ing_data)
        df_req = pd.DataFrame(req_data)
    except Exception as e:
        return {"status": "error", "message": f"数据转换失败: {str(e)}"}

    if '营养素' not in df_req.columns:
        return {"status": "error", "message": "营养文件缺少列: 营养素"}

    nutrient_names = [str(n) for n in df_req['营养素'].tolist()]
    errs = validate_inputs(df_ing, df_req, nutrient_names)
    if errs:
        return {"status": "error", "message": "数据校验失败：\n" + "\n".join(errs)}

    ingredients = [str(n) for n in df_ing['原料名称'].tolist()]
    ing_idx = {n: i for i, n in enumerate(ingredients)}
    nut_idx = {n: j for j, n in enumerate(nutrient_names)}

    price_base = np.array([parse_numeric(v, 0) for v in df_ing['价格']])
    lb_ing_base = np.array([parse_numeric(v, 0) for v in df_ing['用量下限%']]) / 100.0
    ub_ing_base = np.array([parse_numeric(v, 100) for v in df_ing['用量上限%']]) / 100.0

    lb_nut_base = np.array([parse_numeric(v, 0) for v in df_req['下限']])
    ub_nut_base = np.array([parse_numeric(v, 1e9) for v in df_req['上限']])

    content = np.zeros((len(ingredients), len(nutrient_names)))
    for j, nut in enumerate(nutrient_names):
        if nut in df_ing.columns:
            col = df_ing[nut].tolist()
            for i in range(len(ingredients)):
                content[i, j] = parse_numeric(col[i], 0)

    solver, solver_name = get_solver()
    if solver is None:
        return {
            "status": "error",
            "message": "未找到可用的求解器。请安装 highspy：pip install highspy",
        }

    # 构建模型
    model = pyo.ConcreteModel()
    model.x = pyo.Var(ingredients)
    model.price = pyo.Param(ingredients, initialize=lambda m, i: float(price_base[ing_idx[i]]), mutable=True)
    model.lb_ing = pyo.Param(ingredients, initialize=lambda m, i: float(lb_ing_base[ing_idx[i]]), mutable=True)
    model.ub_ing = pyo.Param(ingredients, initialize=lambda m, i: float(ub_ing_base[ing_idx[i]]), mutable=True)
    model.lb_nut = pyo.Param(nutrient_names, initialize=lambda m, j: float(lb_nut_base[nut_idx[j]]), mutable=True)
    model.ub_nut = pyo.Param(nutrient_names, initialize=lambda m, j: float(ub_nut_base[nut_idx[j]]), mutable=True)

    model.c_ing_lb = pyo.Constraint(ingredients, rule=lambda m, i: m.x[i] >= m.lb_ing[i])
    model.c_ing_ub = pyo.Constraint(ingredients, rule=lambda m, i: m.x[i] <= m.ub_ing[i])

    def nut_rule_lb(m, j):
        return sum(content[ing_idx[i], nut_idx[j]] * m.x[i] for i in ingredients) >= m.lb_nut[j]

    def nut_rule_ub(m, j):
        return sum(content[ing_idx[i], nut_idx[j]] * m.x[i] for i in ingredients) <= m.ub_nut[j]

    model.c_nut_lb = pyo.Constraint(nutrient_names, rule=nut_rule_lb)
    model.c_nut_ub = pyo.Constraint(nutrient_names, rule=nut_rule_ub)
    model.c_total = pyo.Constraint(expr=sum(model.x[i] for i in ingredients) == 1)
    model.cost = pyo.Objective(expr=sum(model.price[i] * model.x[i] for i in ingredients), sense=pyo.minimize)
    model.dual = Suffix(direction=Suffix.IMPORT)

    def solve_attempt():
        try:
            # 清理旧后缀，避免读取到陈旧数据
            try:
                for k in list(model.dual.keys()):
                    del model.dual[k]
            except Exception:
                pass
                
            # 强制不加载解，防止无解时抛出异常导致日志刷屏
            r = solver.solve(model, tee=False, load_solutions=False)
            
            # 传统接口的状态检查
            term = getattr(getattr(r, 'solver', None), 'termination_condition', None)
            
            if str(term).lower() == 'optimal':
                # 只有在最优时才加载解
                model.solutions.load_from(r)
                return True
            return False
        except Exception:
            # 在二分查找探测边界时，无解是正常现象，直接静默返回 False
            return False

    if not solve_attempt():
        return {
            "status": "error",
            "message": "无最优解。请检查营养指标或原料限制是否矛盾。\n(提示：通常是某项营养素下限设得过高，或者某项原料用量上限设得过低导致)",
        }

    cost_base = pyo.value(model.cost)
    x_base = {i: pyo.value(model.x[i]) for i in ingredients}

    def safe_dual(constraint_obj):
        try:
            # 传统 highs 会写入 model.dual 后缀
            v = model.dual.get(constraint_obj, 0.0)
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    dual_ing_lb = {i: safe_dual(model.c_ing_lb[i]) for i in ingredients}
    dual_ing_ub = {i: safe_dual(model.c_ing_ub[i]) for i in ingredients}
    dual_nut_lb = {j: safe_dual(model.c_nut_lb[j]) for j in nutrient_names}
    dual_nut_ub = {j: safe_dual(model.c_nut_ub[j]) for j in nutrient_names}

    rc_base = {}
    is_fixed = {}
    for i in ingredients:
        idx = ing_idx[i]
        if abs(ub_ing_base[idx] - lb_ing_base[idx]) < 1e-5:
            is_fixed[i] = True
            rc_base[i] = 0.0
        else:
            is_fixed[i] = False
            if abs(x_base[i] - lb_ing_base[idx]) < 1e-5:
                rc_base[i] = dual_ing_lb[i]
            elif abs(x_base[i] - ub_ing_base[idx]) < 1e-5:
                rc_base[i] = dual_ing_ub[i]
            else:
                rc_base[i] = 0.0

    # 灵敏度分析容差放宽
    def basis_unchanged():
        for i in ingredients:
            bv = x_base[i]
            cv = pyo.value(model.x[i])
            tol = max(1e-3, abs(bv) * 0.05)
            if abs(cv - bv) > tol:
                return False
        return True

    def make_dual_check(constraint_obj, base_dual):
        def check():
            v = safe_dual(constraint_obj)
            if abs(base_dual) < 1e-4:
                return abs(v) < 0.1
            return abs(v - base_dual) <= max(0.1, abs(base_dual) * 0.05)
        return check

    def binary_search_bound(param, base_val, lower_limit, upper_limit, direction, check_func):
        step = max(abs(base_val) * 0.01, 0.01)

        def restore():
            param.set_value(base_val)
            solve_attempt()

        if direction == 'increase':
            feasible, infeasible = base_val, None
            guard = 0
            while guard < 200:
                guard += 1
                trial = base_val + step
                if trial >= upper_limit:
                    trial = upper_limit
                    param.set_value(trial)
                    ok = solve_attempt() and check_func()
                    if not ok:
                        infeasible = trial
                        break
                    restore()
                    if upper_limit < MAX_BOUND:
                        return trial - base_val
                    return None
                param.set_value(trial)
                if not solve_attempt() or not check_func():
                    infeasible = trial
                    break
                feasible = trial
                step *= 2

            if infeasible is None or infeasible <= feasible + 1e-10:
                restore()
                return 0.0

            guard = 0
            while infeasible - feasible > max(0.0001, abs(infeasible) * 1e-5) and guard < 100:
                guard += 1
                mid = (feasible + infeasible) / 2
                param.set_value(mid)
                if solve_attempt() and check_func():
                    feasible = mid
                else:
                    infeasible = mid
            restore()
            return feasible - base_val

        else:  # decrease
            feasible, infeasible = base_val, None
            guard = 0
            while guard < 200:
                guard += 1
                trial = base_val - step
                if trial <= lower_limit:
                    trial = lower_limit
                    param.set_value(trial)
                    ok = solve_attempt() and check_func()
                    if not ok:
                        infeasible = trial
                        break
                    restore()
                    return base_val - trial
                param.set_value(trial)
                if not solve_attempt() or not check_func():
                    infeasible = trial
                    break
                feasible = trial
                step *= 2

            if infeasible is None or feasible <= infeasible + 1e-10:
                restore()
                return 0.0

            guard = 0
            while feasible - infeasible > max(0.0001, abs(feasible) * 1e-5) and guard < 100:
                guard += 1
                mid = (feasible + infeasible) / 2
                param.set_value(mid)
                if solve_attempt() and check_func():
                    feasible = mid
                else:
                    infeasible = mid
            restore()
            return base_val - feasible

    def _fmt_pct(v, is_fixed_flag):
        if is_fixed_flag:
            return "-"
        if v is None:
            return "∞"
        return round(float(v) * 100, 4)

    def _fmt_raw(v):
        if v is None:
            return "∞"
        return round(float(v), 4)

    # ===================== 原料灵敏度 =====================
    result_recipe = []
    for i, ing in enumerate(ingredients):
        idx = ing_idx[ing]
        qty = x_base[ing] * 100

        if is_fixed[ing]:
            ing_inc_lb = ing_dec_lb = ing_inc_ub = ing_dec_ub = 0.0
            dual_ing_lb_val = dual_ing_ub_val = 0.0
            price_lower_disp = "-"
            price_upper_disp = "-"
        else:
            p_inc = binary_search_bound(model.price[ing], price_base[idx], 0, MAX_BOUND, 'increase', basis_unchanged)
            p_dec = binary_search_bound(model.price[ing], price_base[idx], 0, MAX_BOUND, 'decrease', basis_unchanged)
            price_lower_disp = "0" if p_dec is None else round(float(price_base[idx] - p_dec), 2)
            if p_inc is None:
                price_upper_disp = "∞"
            else:
                upper_val = price_base[idx] + p_inc
                price_upper_disp = "∞" if upper_val >= MAX_BOUND else round(float(upper_val), 2)

            check_lb = make_dual_check(model.c_ing_lb[ing], dual_ing_lb[ing])
            check_ub = make_dual_check(model.c_ing_ub[ing], dual_ing_ub[ing])
            ing_inc_lb = binary_search_bound(model.lb_ing[ing], lb_ing_base[idx], 0, 1.0, 'increase', check_lb)
            ing_dec_lb = binary_search_bound(model.lb_ing[ing], lb_ing_base[idx], 0, 1.0, 'decrease', check_lb)
            ing_inc_ub = binary_search_bound(model.ub_ing[ing], ub_ing_base[idx], 0, 1.0, 'increase', check_ub)
            ing_dec_ub = binary_search_bound(model.ub_ing[ing], ub_ing_base[idx], 0, 1.0, 'decrease', check_ub)
            dual_ing_lb_val = dual_ing_lb[ing]
            dual_ing_ub_val = dual_ing_ub[ing]

        result_recipe.append({
            "原料名称": ing,
            "是否固定": is_fixed[ing],
            "结果%": round(qty, 4),
            "递减成本": round(rc_base[ing] / 100, 4) if not is_fixed[ing] else None,
            "当前价格": float(price_base[idx]),
            "价格下限": price_lower_disp,
            "价格上限": price_upper_disp,
            "下限影子价格": round(float(dual_ing_lb_val) / 100, 4) if not is_fixed[ing] else None,
            "上限影子价格": round(float(dual_ing_ub_val) / 100, 4) if not is_fixed[ing] else None,
            "下限允许增加": _fmt_pct(ing_inc_lb, is_fixed[ing]),
            "下限允许减少": _fmt_pct(ing_dec_lb, is_fixed[ing]),
            "上限允许增加": _fmt_pct(ing_inc_ub, is_fixed[ing]),
            "上限允许减少": _fmt_pct(ing_dec_ub, is_fixed[ing]),
        })

    # ===================== 营养灵敏度 =====================
    result_nutrients = []
    x_vec = np.array([x_base[i] for i in ingredients])

    for nut in nutrient_names:
        j = nut_idx[nut]
        actual = float(np.dot(content[:, j], x_vec))

        check_lb = make_dual_check(model.c_nut_lb[nut], dual_nut_lb[nut])
        check_ub = make_dual_check(model.c_nut_ub[nut], dual_nut_ub[nut])

        n_inc_lb = binary_search_bound(model.lb_nut[nut], lb_nut_base[j], 0, MAX_BOUND, 'increase', check_lb)
        n_dec_lb = binary_search_bound(model.lb_nut[nut], lb_nut_base[j], 0, MAX_BOUND, 'decrease', check_lb)
        n_inc_ub = binary_search_bound(model.ub_nut[nut], ub_nut_base[j], 0, MAX_BOUND, 'increase', check_ub)
        n_dec_ub = binary_search_bound(model.ub_nut[nut], ub_nut_base[j], 0, MAX_BOUND, 'decrease', check_ub)

        ub_disp = "∞" if ub_nut_base[j] >= 1e8 else float(ub_nut_base[j])
        status = "达标" if (lb_nut_base[j] - 1e-4) <= actual <= (ub_nut_base[j] + 1e-4) else "不达标"

        result_nutrients.append({
            "营养素": nut,
            "结果": round(actual, 4),
            "下限": float(lb_nut_base[j]),
            "上限": ub_disp,
            "状态": status,
            "下界影子价格": round(float(dual_nut_lb[nut]), 4),
            "下界允许增加": _fmt_raw(n_inc_lb),
            "下界允许减少": _fmt_raw(n_dec_lb),
            "上界影子价格": round(float(dual_nut_ub[nut]), 4),
            "上界允许增加": _fmt_raw(n_inc_ub),
            "上界允许减少": _fmt_raw(n_dec_ub),
        })

    return {
        "status": "success",
        "total_cost": round(float(cost_base), 2),
        "recipe": result_recipe,
        "nutrients": result_nutrients,
        "solver": solver_name,
    }


@app.post("/api/solve")
def solve_formula(req: SolveRequest):
    try:
        result = run_lp(req.ingredients, req.requirements)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"服务器错误: {str(e)}"}


# ================== Excel 导出 ==================
@app.post("/api/export")
def export_excel(req: SolveRequest):
    try:
        result = run_lp(req.ingredients, req.requirements)
        if result["status"] != "success":
            return Response(content=result.get("message", "导出失败"), media_type="text/plain", status_code=400)

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_summary = pd.DataFrame({
                '项目': ['最优总成本'],
                '数值': [result['total_cost']],
                '单位': ['元/吨'],
            })
            df_recipe = pd.DataFrame(result['recipe'])
            df_recipe = df_recipe.fillna('-').astype(object)
            df_recipe_export = df_recipe[['原料名称', '结果%', '当前价格', '价格下限', '价格上限']].copy()

            df_summary.to_excel(writer, sheet_name="配方与成本", index=False, startrow=0)
            df_recipe_export.to_excel(writer, sheet_name="配方与成本", index=False, startrow=3)
            _beautify_header(writer, "配方与成本", df_summary, start_row=0)
            _beautify_header(writer, "配方与成本", df_recipe_export, start_row=3)
            _autosize_columns(writer, "配方与成本", [df_summary, df_recipe_export])

            df_nut = pd.DataFrame(result['nutrients'])
            df_nut = df_nut.fillna('-').astype(object)
            df_nut_export = df_nut[['营养素', '结果', '下限', '上限', '状态', '下界影子价格', '上界影子价格']].copy()
            df_nut_export.to_excel(writer, sheet_name="营养分析", index=False)
            _beautify_header(writer, "营养分析", df_nut_export)
            _autosize_columns(writer, "营养分析", [df_nut_export])

            df_sens_nut = df_nut[['营养素', '下界允许增加', '下界允许减少', '上界允许增加', '上界允许减少']].copy()
            df_sens_nut.to_excel(writer, sheet_name="营养灵敏度", index=False)
            _beautify_header(writer, "营养灵敏度", df_sens_nut)
            _autosize_columns(writer, "营养灵敏度", [df_sens_nut])

            df_sens_ing = df_recipe[[
                '原料名称', '递减成本', '下限影子价格', '下限允许增加', '下限允许减少',
                '上限影子价格', '上限允许增加', '上限允许减少',
            ]].copy()
            df_sens_ing.to_excel(writer, sheet_name="原料灵敏度", index=False)
            _beautify_header(writer, "原料灵敏度", df_sens_ing)
            _autosize_columns(writer, "原料灵敏度", [df_sens_ing])

        excel_buffer.seek(0)
        return Response(
            content=excel_buffer.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=feed_formula_report.xlsx"},
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(content=f"导出错误: {str(e)}", media_type="text/plain", status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)