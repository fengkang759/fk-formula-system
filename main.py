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


# ================== 核心求解（highspy 矩阵接口 + getRanging） ==================
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
    ing_idx = {name: i for i, name in enumerate(ingredients)}
    nut_idx = {name: j for j, name in enumerate(nutrient_names)}

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

    # 固定变量判断
    is_fixed = np.abs(ub_ing - lb_ing) < 1e-5

    # ---------- 构建 highspy 模型 ----------
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)

    h.addCols(n, price_base, lb_ing, ub_ing, 0, 0, 0, 0)

    # 营养下限行
    for j in range(m):
        idx = []
        val = []
        for i in range(n):
            if content[i, j] != 0.0:
                idx.append(i)
                val.append(content[i, j])
        h.addRow(lb_nut[j], INF, len(idx), np.array(idx, dtype=np.int32), np.array(val, dtype=np.double))

    # 营养上限行
    for j in range(m):
        idx = []
        val = []
        for i in range(n):
            if content[i, j] != 0.0:
                idx.append(i)
                val.append(content[i, j])
        h.addRow(-INF, ub_nut[j], len(idx), np.array(idx, dtype=np.int32), np.array(val, dtype=np.double))

    # 总用量 = 1
    all_idx = list(range(n))
    all_val = [1.0] * n
    h.addRow(1.0, 1.0, n, np.array(all_idx, dtype=np.int32), np.array(all_val, dtype=np.double))

    # ---------- 求解 ----------
    h.run()
    status = h.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        return {
            "status": "error",
            "message": "无最优解。请检查营养指标或原料限制是否矛盾。\n"
                       "(提示：通常是某项营养素下限设得过高，或者某项原料用量上限设得过低导致)",
        }

    solution = h.getSolution()
    info = h.getInfo()

    x_val = np.array(solution.col_value, dtype=np.double)
    row_dual = np.array(solution.row_dual, dtype=np.double)
    col_dual = np.array(solution.col_dual, dtype=np.double)

    cost_base = float(info.objective_function_value)

    status_r, ranging = h.getRanging()
    if status_r != highspy.HighsStatus.kOk:
        return {"status": "error", "message": "灵敏度分析获取失败（getRanging 返回非 Ok）"}

    # ---------- 格式化与安全读取辅助 ----------
    def safe_idx(arr, idx, default=0.0):
        try:
            if idx < len(arr):
                return arr[idx]
        except Exception:
            pass
        return default

    def _fmt_pct(v):
        if v is None or abs(v) >= INF / 2:
            return "∞"
        return round(float(v) * 100, 4)

    def _fmt_raw(v):
        if v is None or abs(v) >= INF / 2:
            return "∞"
        return round(float(v), 4)

    # 【优化】用 -∞ 和 ∞ 表达价格的界限
    def _price_bound(base, ranged_value, is_lower=True):
        if abs(ranged_value) >= INF / 2:
            return "-∞" if is_lower else "∞"
        return round(float(ranged_value), 2)

    # ===================== 原料结果 + 灵敏度 =====================
    result_recipe = []
    for i, ing in enumerate(ingredients):
        qty = float(x_val[i]) * 100.0
        fixed = bool(is_fixed[i])

        if fixed:
            result_recipe.append({
                "原料名称": ing,
                "是否固定": True,
                "结果%": round(qty, 4),
                "递减成本": None,
                "当前价格": float(price_base[i]),
                "价格下限": "-",
                "价格上限": "-",
                "下限影子价格": None,
                "上限影子价格": None,
                "下限允许增加": "-",
                "下限允许减少": "-",
                "上限允许增加": "-",
                "上限允许减少": "-",
            })
            continue

        cost_dn_val = safe_idx(ranging.col_cost_dn.value_, i, price_base[i])
        cost_up_val = safe_idx(ranging.col_cost_up.value_, i, price_base[i])
        
        price_lower = _price_bound(price_base[i], cost_dn_val, is_lower=True)
        price_upper = _price_bound(price_base[i], cost_up_val, is_lower=False)

        lb_dn_val = safe_idx(ranging.col_bound_dn.value_, i, -INF) if lb_ing[i] > -INF / 2 else -INF
        lb_up_val = safe_idx(ranging.col_bound_up.value_, i, INF) if lb_ing[i] > -INF / 2 else INF
        ub_dn_val = safe_idx(ranging.col_bound_dn.value_, i, -INF) if ub_ing[i] < INF / 2 else -INF
        ub_up_val = safe_idx(ranging.col_bound_up.value_, i, INF) if ub_ing[i] < INF / 2 else INF

        if lb_ing[i] <= 1e-9:
            ing_dec_lb_disp = "0"
        elif abs(lb_dn_val) >= INF / 2:
            ing_dec_lb_disp = "∞"
        else:
            ing_dec_lb_disp = _fmt_pct(max(0.0, lb_ing[i] - lb_dn_val))

        if abs(lb_up_val) >= INF / 2:
            ing_inc_lb_disp = "∞"
        else:
            ing_inc_lb_disp = _fmt_pct(max(0.0, lb_up_val - lb_ing[i]))

        if ub_ing[i] >= 1.0 - 1e-9:
            ing_inc_ub_disp = "0"
        elif abs(ub_up_val) >= INF / 2:
            ing_inc_ub_disp = "∞"
        else:
            ing_inc_ub_disp = _fmt_pct(max(0.0, ub_up_val - ub_ing[i]))

        if abs(ub_dn_val) >= INF / 2:
            ing_dec_ub_disp = "∞"
        else:
            ing_dec_ub_disp = _fmt_pct(max(0.0, ub_ing[i] - ub_dn_val))

        col_dual_val = safe_idx(col_dual, i, 0.0)

        result_recipe.append({
            "原料名称": ing,
            "是否固定": False,
            "结果%": round(qty, 4),
            "递减成本": round(float(col_dual_val) / 100.0, 4),
            "当前价格": float(price_base[i]),
            "价格下限": price_lower,
            "价格上限": price_upper,
            "下限影子价格": 0.0,
            "上限影子价格": 0.0,
            "下限允许增加": ing_inc_lb_disp,
            "下限允许减少": ing_dec_lb_disp,
            "上限允许增加": ing_inc_ub_disp,
            "上限允许减少": ing_dec_ub_disp,
        })

    # 修正原料上下限影子价格判断逻辑，增加距离比较，防止由于浮点误差或基退化导致判断错误
    for entry in result_recipe:
        if entry["是否固定"]:
            continue
        i = ing_idx[entry["原料名称"]]
        xv = float(x_val[i])
        cd = float(safe_idx(col_dual, i, 0.0))
        dist_lb = abs(xv - lb_ing[i])
        dist_ub = abs(xv - ub_ing[i])
        
        if dist_lb < 1e-4 and dist_lb <= dist_ub:
            entry["下限影子价格"] = round(cd / 100.0, 4)
            entry["上限影子价格"] = 0.0
        elif dist_ub < 1e-4:
            entry["下限影子价格"] = 0.0
            entry["上限影子价格"] = round(cd / 100.0, 4)
        else:
            entry["下限影子价格"] = 0.0
            entry["上限影子价格"] = 0.0

    # ===================== 营养结果 + 灵敏度 =====================
    result_nutrients = []
    for j, nut in enumerate(nutrient_names):
        actual = float(np.dot(content[:, j], x_val))
        lb_row = j
        ub_row = m + j

        lb_dn_val = safe_idx(ranging.row_bound_dn.value_, lb_row, -INF)
        lb_up_val = safe_idx(ranging.row_bound_up.value_, lb_row, INF)
        ub_dn_val = safe_idx(ranging.row_bound_dn.value_, ub_row, -INF)
        ub_up_val = safe_idx(ranging.row_bound_up.value_, ub_row, INF)

        row_dual_lb = safe_idx(row_dual, lb_row, 0.0)
        row_dual_ub = safe_idx(row_dual, ub_row, 0.0)

        if abs(lb_up_val) >= INF / 2:
            n_inc_lb = "∞"
        else:
            n_inc_lb = _fmt_raw(max(0.0, lb_up_val - lb_nut[j]))

        if lb_nut[j] <= 1e-9:
            n_dec_lb = "0"
        elif abs(lb_dn_val) >= INF / 2:
            n_dec_lb = "∞"
        else:
            n_dec_lb = _fmt_raw(max(0.0, lb_nut[j] - lb_dn_val))

        if ub_nut[j] >= 1e8:
            n_inc_ub = "0"
        elif abs(ub_up_val) >= INF / 2:
            n_inc_ub = "∞"
        else:
            n_inc_ub = _fmt_raw(max(0.0, ub_up_val - ub_nut[j]))

        if abs(ub_dn_val) >= INF / 2:
            n_dec_ub = "∞"
        else:
            n_dec_ub = _fmt_raw(max(0.0, ub_nut[j] - ub_dn_val))

        ub_disp = "∞" if ub_nut[j] >= 1e8 else float(ub_nut[j])
        status_str = "达标" if (lb_nut[j] - 1e-4) <= actual <= (ub_nut[j] + 1e-4) else "不达标"

        result_nutrients.append({
            "营养素": nut,
            "结果": round(actual, 4),
            "下限": float(lb_nut[j]),
            "上限": ub_disp,
            "状态": status_str,
            "下界影子价格": round(float(row_dual_lb), 4),
            "下界允许增加": n_inc_lb,
            "下界允许减少": n_dec_lb,
            "上界影子价格": round(float(row_dual_ub), 4),
            "上界允许增加": n_inc_ub,
            "上界允许减少": n_dec_ub,
        })

    return {
        "status": "success",
        "total_cost": round(cost_base, 2),
        "recipe": result_recipe,
        "nutrients": result_nutrients,
        "solver": "highs (highspy matrix)",
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