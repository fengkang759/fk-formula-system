from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import highspy
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

MAX_UPLOAD_SIZE = 10 * 1024 * 1024
VALID_RECIPE_RE = re.compile(r'^[\w\u4e00-\u9fa5\-]+$')
INF = highspy.kHighsInf


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


# ================== 参数扫描：RHS 允许变化 ==================
def scan_rhs(h, row_idx, side, current_lo, current_hi, base_obj, base_dual,
             direction, base_basis=None, max_iters=60):
    """
    扫描行 row_idx 的 RHS 在 direction 方向可以变化多少，基不变。
    side='lower': 改下界 current_lo；side='upper': 改上界 current_hi。
    direction: +1 增大，-1 减小。
    base_basis: 原始最优基（用于 warm start，防止退化基下 HiGHS 主动重选基）。
    返回：允许变化量（绝对值），INF 表示无界。
    """
    if side == 'lower':
        current_rhs = current_lo

        def set_rhs(v):
            h.changeRowBounds(row_idx, v, current_hi)
    else:
        current_rhs = current_hi

        def set_rhs(v):
            h.changeRowBounds(row_idx, current_lo, v)

    def is_same_basis(v):
        set_rhs(v)
        if base_basis is not None:
            try:
                h.setBasis(base_basis)
            except Exception:
                pass
        h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return False
        try:
            new_dual = float(h.getSolution().row_dual[row_idx])
            new_obj = float(h.getInfo().objective_function_value)
        except Exception:
            return False
        delta = v - current_rhs
        expected = base_obj + base_dual * delta
        tol = 1e-6 * max(1.0, abs(base_obj))
        if abs(new_dual - base_dual) > 1e-5:
            return False
        if abs(new_obj - expected) > tol:
            return False
        return True

    init_step = max(abs(current_rhs) * 0.01, 1e-4)
    last_good = current_rhs
    step = init_step

    found = False
    probe = current_rhs
    for _ in range(45):
        probe = current_rhs + direction * step
        if not is_same_basis(probe):
            found = True
            break
        last_good = probe
        step *= 2
        if step > 1e12:
            break

    if not found:
        set_rhs(current_rhs)
        if base_basis is not None:
            try:
                h.setBasis(base_basis)
            except Exception:
                pass
        h.run()
        return INF

    lo = min(last_good, probe)
    hi = max(last_good, probe)

    for _ in range(max_iters):
        mid = (lo + hi) / 2
        same = is_same_basis(mid)
        if direction > 0:
            if same:
                lo = mid
            else:
                hi = mid
        else:
            if same:
                hi = mid
            else:
                lo = mid
        if (hi - lo) < 1e-9 * max(1.0, abs(current_rhs)):
            break

    set_rhs(current_rhs)
    if base_basis is not None:
        try:
            h.setBasis(base_basis)
        except Exception:
            pass
    h.run()

    if direction > 0:
        return max(0.0, lo - current_rhs)
    else:
        return max(0.0, current_rhs - hi)


def row_sensitivity(h, row_idx, side, bound_val, actual, base_obj, base_dual,
                    base_basis=None):
    """
    计算一行的 (允许增量, 允许减量)。
    side='lower': RHS=bound_val 是下界
    side='upper': RHS=bound_val 是上界
    """
    if side == 'lower':
        is_active = abs(actual - bound_val) < 1e-5
        if is_active:
            inc = scan_rhs(h, row_idx, 'lower', bound_val, INF, base_obj, base_dual,
                           +1, base_basis)
            dec = scan_rhs(h, row_idx, 'lower', bound_val, INF, base_obj, base_dual,
                           -1, base_basis)
        else:
            inc = max(0.0, actual - bound_val)
            dec = INF
        return inc, dec
    else:
        is_active = abs(actual - bound_val) < 1e-5
        if is_active:
            inc = scan_rhs(h, row_idx, 'upper', -INF, bound_val, base_obj, base_dual,
                           +1, base_basis)
            dec = scan_rhs(h, row_idx, 'upper', -INF, bound_val, base_obj, base_dual,
                           -1, base_basis)
        else:
            inc = INF
            dec = max(0.0, bound_val - actual)
        return inc, dec


# ================== 辅助模型：变量带上下界，获取递减成本 ==================
def compute_reduced_costs(n, price_base, lb_ing, ub_ing, content, lb_nut, ub_nut, m):
    """
    构建变量带上下界的辅助模型（与 Excel 规划求解口径一致），求解获取 col_dual
    （即递减成本）。
    """
    try:
        h_aux = highspy.Highs()
        h_aux.setOptionValue("output_flag", False)
        h_aux.addCols(n, price_base, lb_ing, ub_ing, 0, 0, 0, 0)

        for j in range(m):
            idx_list = [i for i in range(n) if content[i, j] != 0.0]
            val_list = [content[i, j] for i in idx_list]
            h_aux.addRow(lb_nut[j], INF, len(idx_list),
                         np.array(idx_list, dtype=np.int32),
                         np.array(val_list, dtype=np.double))

        for j in range(m):
            idx_list = [i for i in range(n) if content[i, j] != 0.0]
            val_list = [content[i, j] for i in idx_list]
            h_aux.addRow(-INF, ub_nut[j], len(idx_list),
                         np.array(idx_list, dtype=np.int32),
                         np.array(val_list, dtype=np.double))

        h_aux.addRow(1.0, 1.0, n, np.arange(n, dtype=np.int32), np.ones(n, dtype=np.double))
        h_aux.run()
        if h_aux.getModelStatus() == highspy.HighsModelStatus.kOptimal:
            return np.array(h_aux.getSolution().col_dual, dtype=np.double)
    except Exception:
        pass
    return np.zeros(n)


# ================== 核心求解 ==================
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
    n = len(ingredients)
    m = len(nutrient_names)

    price_base = np.array([parse_numeric(v, 0) for v in df_ing['价格']], dtype=np.double)
    lb_ing = np.array([parse_numeric(v, 0) for v in df_ing['用量下限%']], dtype=np.double) / 100.0
    ub_ing = np.array([parse_numeric(v, 100) for v in df_ing['用量上限%']], dtype=np.double) / 100.0

    lb_nut = np.array([parse_numeric(v, 0) for v in df_req['下限']], dtype=np.double)
    ub_nut = np.array([parse_numeric(v, 1e9) for v in df_req['上限']], dtype=np.double)

    content = np.zeros((n, m), dtype=np.double)
    for j, nut in enumerate(nutrient_names):
        if nut in df_ing.columns:
            col = df_ing[nut].tolist()
            for i in range(n):
                content[i, j] = parse_numeric(col[i], 0)

    # ---------- 主模型：变量 [0, INF)，上下界作为约束行 ----------
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.addCols(n, price_base, np.zeros(n), np.full(n, INF), 0, 0, 0, 0)

    row_idx = 0
    h.addRow(1.0, 1.0, n, np.arange(n, dtype=np.int32), np.ones(n, dtype=np.double))
    row_idx += 1

    ing_ub_rows = [None] * n
    for i in range(n):
        if ub_ing[i] > lb_ing[i] + 1e-9:
            h.addRow(-INF, ub_ing[i], 1, np.array([i], dtype=np.int32), np.array([1.0], dtype=np.double))
            ing_ub_rows[i] = row_idx
            row_idx += 1

    ing_lb_rows = [None] * n
    for i in range(n):
        h.addRow(lb_ing[i], INF, 1, np.array([i], dtype=np.int32), np.array([1.0], dtype=np.double))
        ing_lb_rows[i] = row_idx
        row_idx += 1

    nut_lb_rows = [None] * m
    for j in range(m):
        idx_list = [i for i in range(n) if content[i, j] != 0.0]
        val_list = [content[i, j] for i in idx_list]
        h.addRow(lb_nut[j], INF, len(idx_list),
                 np.array(idx_list, dtype=np.int32), np.array(val_list, dtype=np.double))
        nut_lb_rows[j] = row_idx
        row_idx += 1

    nut_ub_rows = [None] * m
    for j in range(m):
        idx_list = [i for i in range(n) if content[i, j] != 0.0]
        val_list = [content[i, j] for i in idx_list]
        h.addRow(-INF, ub_nut[j], len(idx_list),
                 np.array(idx_list, dtype=np.int32), np.array(val_list, dtype=np.double))
        nut_ub_rows[j] = row_idx
        row_idx += 1

    # ---------- 求解主模型 ----------
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return {
            "status": "error",
            "message": "无最优解。请检查营养指标或原料限制是否矛盾。\n"
                       "(提示：通常是某项营养素下限设得过高，或者某项原料用量上限设得过低导致)",
        }

    solution = h.getSolution()
    info = h.getInfo()

    x_val = np.array(solution.col_value, dtype=np.double)
    row_dual = np.array(solution.row_dual, dtype=np.double)
    base_obj = float(info.objective_function_value)

    try:
        base_basis = h.getBasis()
    except Exception:
        base_basis = None

    status_r, ranging = h.getRanging()
    if status_r != highspy.HighsStatus.kOk:
        return {"status": "error", "message": "灵敏度分析获取失败（getRanging 返回非 Ok）"}

    # ---------- 辅助模型：获取递减成本（变量带上下界口径） ----------
    col_dual_aux = compute_reduced_costs(
        n, price_base, lb_ing, ub_ing, content, lb_nut, ub_nut, m
    )

    # ---------- 格式化辅助 ----------
    def safe_idx(arr, idx, default=0.0):
        try:
            if idx is not None and idx < len(arr):
                return arr[idx]
        except Exception:
            pass
        return default

    def _is_inf(v):
        if v is None:
            return True
        try:
            return np.isinf(v) or abs(float(v)) >= INF / 2
        except (TypeError, ValueError):
            return True

    def _fmt_pct_frac(v):
        if v is None:
            return "-"
        if _is_inf(v):
            return "∞"
        try:
            return round(float(v) * 100, 4)
        except (TypeError, ValueError):
            return "-"

    def _fmt_raw(v):
        if v is None:
            return "-"
        if _is_inf(v):
            return "∞"
        try:
            return round(float(v), 4)
        except (TypeError, ValueError):
            return "-"

    def _price_bound(ranged_value, is_lower=True):
        if _is_inf(ranged_value):
            return "-∞" if is_lower else "∞"
        try:
            return round(float(ranged_value), 2)
        except (TypeError, ValueError):
            return "-∞" if is_lower else "∞"

    # ===================== 原料结果 + 灵敏度 =====================
    result_recipe = []
    for i, ing in enumerate(ingredients):
        qty = float(x_val[i]) * 100.0
        lb_i = float(lb_ing[i])
        ub_i = float(ub_ing[i])
        xi = float(x_val[i])
        is_fixed = (ub_i - lb_i) < 1e-9

        # 固定变量：所有灵敏度列全部显示 "-"
        if is_fixed:
            result_recipe.append({
                "原料名称": ing,
                "结果%": round(qty, 4),
                "递减成本": "-",
                "当前价格": float(price_base[i]),
                "价格下限": "-",
                "价格上限": "-",
                "下限影子价格": "-",
                "下限允许增加": "-",
                "下限允许减少": "-",
                "上限影子价格": "-",
                "上限允许增加": "-",
                "上限允许减少": "-",
            })
            continue

        # 【本次修改】递减成本：上界变量置 0（Excel 口径）；其余用辅助模型 col_dual
        if is_fixed:
            reduced_cost = "-"
        elif abs(xi - ub_i) < 1e-6:
            # 活跃在上界的变量，Excel 递减成本显示 0
            reduced_cost = 0.0
        else:
            rc = float(col_dual_aux[i]) if i < len(col_dual_aux) else 0.0
            reduced_cost = 0.0 if abs(rc) < 1e-6 else round(rc, 4)

        # 价格上下限
        cost_dn_val = safe_idx(ranging.col_cost_dn.value_, i, price_base[i])
        cost_up_val = safe_idx(ranging.col_cost_up.value_, i, price_base[i])
        price_lower = _price_bound(cost_dn_val, is_lower=True)
        price_upper = _price_bound(cost_up_val, is_lower=False)

        # 上限约束行
        dual_ub = 0.0
        inc_ub_disp = "-"
        dec_ub_disp = "-"
        ub_r = ing_ub_rows[i]
        if ub_r is not None:
            base_dual_ub_raw = float(row_dual[ub_r]) if abs(xi - ub_i) < 1e-6 else 0.0
            dual_ub = base_dual_ub_raw / 100.0
            if abs(base_dual_ub_raw) > 1e-7 or abs(xi - ub_i) < 1e-5:
                inc_ub, dec_ub = row_sensitivity(h, ub_r, 'upper', ub_i, xi,
                                                 base_obj, base_dual_ub_raw, base_basis)
            else:
                inc_ub, dec_ub = INF, max(0.0, ub_i - xi)
            inc_ub_disp = _fmt_pct_frac(inc_ub)
            dec_ub_disp = _fmt_pct_frac(dec_ub)

        # 下限约束行
        dual_lb = 0.0
        inc_lb_disp = "-"
        dec_lb_disp = "-"
        lb_r = ing_lb_rows[i]
        if lb_r is not None:
            base_dual_lb_raw = float(row_dual[lb_r]) if abs(xi - lb_i) < 1e-6 else 0.0
            dual_lb = base_dual_lb_raw / 100.0
            if abs(base_dual_lb_raw) > 1e-7 or abs(xi - lb_i) < 1e-5:
                inc_lb, dec_lb = row_sensitivity(h, lb_r, 'lower', lb_i, xi,
                                                 base_obj, base_dual_lb_raw, base_basis)
            else:
                inc_lb, dec_lb = max(0.0, xi - lb_i), INF
            inc_lb_disp = _fmt_pct_frac(inc_lb)
            dec_lb_disp = _fmt_pct_frac(dec_lb)

        result_recipe.append({
            "原料名称": ing,
            "结果%": round(qty, 4),
            "递减成本": reduced_cost,
            "当前价格": float(price_base[i]),
            "价格下限": price_lower,
            "价格上限": price_upper,
            "下限影子价格": round(dual_lb, 4),
            "下限允许增加": inc_lb_disp,
            "下限允许减少": dec_lb_disp,
            "上限影子价格": round(dual_ub, 4),
            "上限允许增加": inc_ub_disp,
            "上限允许减少": dec_ub_disp,
        })

    # ===================== 营养结果 + 灵敏度 =====================
    result_nutrients = []
    for j, nut in enumerate(nutrient_names):
        actual = float(np.dot(content[:, j], x_val))
        lb_j = float(lb_nut[j])
        ub_j = float(ub_nut[j])

        lb_r = nut_lb_rows[j]
        ub_r = nut_ub_rows[j]
        base_dual_lb = float(row_dual[lb_r]) if abs(actual - lb_j) < 1e-6 else 0.0
        base_dual_ub = float(row_dual[ub_r]) if abs(actual - ub_j) < 1e-6 else 0.0

        lb_inc, lb_dec = row_sensitivity(h, lb_r, 'lower', lb_j, actual,
                                         base_obj, base_dual_lb, base_basis)
        ub_inc, ub_dec = row_sensitivity(h, ub_r, 'upper', ub_j, actual,
                                         base_obj, base_dual_ub, base_basis)

        ub_disp = "∞" if ub_j >= 1e8 else float(ub_j)
        status_str = "达标" if (lb_j - 1e-4) <= actual <= (ub_j + 1e-4) else "不达标"

        result_nutrients.append({
            "营养素": nut,
            "结果": round(actual, 4),
            "下限": float(lb_j),
            "上限": ub_disp,
            "状态": status_str,
            "下界影子价格": round(float(row_dual[lb_r]), 4),
            "下界允许增加": _fmt_raw(lb_inc),
            "下界允许减少": _fmt_raw(lb_dec),
            "上界影子价格": round(float(row_dual[ub_r]), 4),
            "上界允许增加": _fmt_raw(ub_inc),
            "上界允许减少": _fmt_raw(ub_dec),
        })

    return {
        "status": "success",
        "total_cost": round(base_obj, 2),
        "recipe": result_recipe,
        "nutrients": result_nutrients,
        "solver": "highs (excel-equivalent ranging + aux reduced cost)",
    }


@app.post("/api/solve")
def solve_formula(req: SolveRequest):
    try:
        return run_lp(req.ingredients, req.requirements)
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