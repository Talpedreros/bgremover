# app.py — PNG transparente, UI corporativa verde, progreso por imagen, sin selecciones por defecto
import io, zipfile, tempfile, re
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
import requests

# ===== Colores =====
PRIMARY_BLUE = "#0B62B3"   # header
ACCENT_GREEN = "#16A34A"   # TODOS los botones y dropzones
SB_BG        = "#F6FAFF"

st.set_page_config(page_title="Catálogo: Quitar fondo y redimensionar (PNG)", layout="wide", page_icon="🖼️")

def inject_css():
    st.markdown(f"""
    <style>
      section[data-testid="stSidebar"] {{
        background: {SB_BG} !important; border-right: 1px solid #e5eef9;
      }}
      .hero {{
        background: {PRIMARY_BLUE}; color: #fff; border-radius: 14px; padding: 14px 18px; margin-bottom: 14px;
      }}
      .hero h1 {{ margin: 0 0 6px 0; color: #fff !important; }}
      .hero p  {{ margin: 0; opacity: .95; }}
      /* TODOS los botones en verde, sin depender del hover */
      .stButton > button, button[kind="secondary"], button[kind="primary"] {{
        background: {ACCENT_GREEN} !important; color: #fff !important;
        border: none !important; border-radius: 10px !important; font-weight: 600 !important;
        box-shadow: 0 4px 10px rgba(0,0,0,.06) !important;
      }}
      .stButton > button:hover, button[kind="secondary"]:hover, button[kind="primary"]:hover {{ filter: brightness(.95); }}
      /* Cards de drag&drop VERDES y con textos blancos */
      div[data-testid="stFileUploader"] > section[data-testid="stFileUploaderDropzone"] {{
        background: {ACCENT_GREEN} !important; border: 2px solid {ACCENT_GREEN} !important; color: #fff !important;
        border-radius: 12px !important;
      }}
      div[data-testid="stFileUploader"] > section * {{ color: #fff !important; }}
      /* Botón "Browse files" dentro del dropzone en verde fijo */
      div[data-testid="stFileUploader"] section button {{ background: #0E893D !important; color: #fff !important; }}
    </style>
    """, unsafe_allow_html=True)

inject_css()

st.markdown(f"""
<div class="hero">
  <h1>Catálogo: Quitar fondo y redimensionar (PNG)</h1>
  <p>Arrastra imágenes sueltas, un ZIP o un Excel/CSV con URLs. Salida siempre en <b>PNG</b> (transparente).
  Descargas individuales si son pocas o ZIP si son muchas. Se genera un manifiesto de errores/no encontradas.</p>
</div>
""", unsafe_allow_html=True)

# ===== Quitar fondo (rembg) =====
REMBG_OK = False
try:
    from rembg import remove as rembg_remove, new_session
    SESSION = new_session("isnet-general-use")
    REMBG_OK = True
except Exception:
    REMBG_OK = False

# ===== Utilidades =====
INVALID_RE = re.compile(r"[^\w\-.]+")

def safe_name(name: str) -> str:
    base = INVALID_RE.sub("_", (name or "").strip())
    return base[:150] if len(base) > 150 else (base or "archivo")

def is_image_name(name: str) -> bool:
    return name.lower().endswith((".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"))

def image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGBA")

def ensure_square_canvas(img: Image.Image, size: int) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGBA","LA"): img = img.convert("RGBA")
    w,h = img.size
    scale = min(size/w, size/h) if (w>size or h>size) else 1.0   # no forzar upscale
    nw,nh = max(1,int(round(w*scale))), max(1,int(round(h*scale)))
    if (nw,nh)!=(w,h): img = img.resize((nw,nh), Image.LANCZOS)
    can = Image.new("RGBA",(size,size),(0,0,0,0))
    can.paste(img, ((size-nw)//2,(size-nh)//2), img)
    return can

def download_image(url: str, sess: requests.Session) -> Optional[Image.Image]:
    try:
        r = sess.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=30)
        if r.status_code != 200:
            return None
        return image_from_bytes(r.content)
    except Exception:
        return None

def remove_bg(img: Image.Image, alta_calidad: bool) -> Image.Image:
    if not REMBG_OK:
        return img
    if alta_calidad:
        return rembg_remove(
            img,
            session=SESSION,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_structure_size=10,
            alpha_matting_base_size=1000,
            post_process_mask=True
        )
    return rembg_remove(img, session=SESSION)

def make_zip(dir_path: Path) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in dir_path.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(dir_path)))
    mem.seek(0)
    return mem.read()

# ===== Sidebar =====
st.sidebar.header("Entrada")
f_imgs = st.sidebar.file_uploader("Imágenes sueltas", type=["png","jpg","jpeg","webp","bmp","tif","tiff"], accept_multiple_files=True)
f_zip  = st.sidebar.file_uploader("ZIP con imágenes", type=["zip"])
f_xl   = st.sidebar.file_uploader("Excel/CSV con URL (y opcional SKU)", type=["xlsx","csv"])

st.sidebar.header("Opciones")
sizes = st.sidebar.multiselect(
    "Tamaños destino (px)",
    [2000,1000,600,450,250],
    default=[],              # <<< sin selección por defecto
    help="Selecciona 1 o más tamaños cuadrados"
)
usar_quitar_fondo = st.sidebar.checkbox("Quitar fondo", value=True)
alta_calidad = st.sidebar.checkbox("Alta calidad de recorte", value=True, disabled=(not usar_quitar_fondo))
forzar_2000 = st.sidebar.checkbox("Forzar /img/2000/ en URLs (Excel/CSV)", value=True)

# ===== Trabajo / salida =====
work = Path(tempfile.mkdtemp(prefix="cat_"))
out_dir = work / "salida"
out_dir.mkdir(parents=True, exist_ok=True)
mani_rows = []
generated_files: list[Path] = []

# ---- helpers de progreso ----
def count_zip_images(file) -> int:
    try:
        with zipfile.ZipFile(file) as z:
            return sum(1 for n in z.namelist() if is_image_name(n))
    except Exception:
        return 0

def count_excel_rows(upload) -> int:
    try:
        if upload.name.lower().endswith(".csv"):
            df = pd.read_csv(upload)
        else:
            df = pd.read_excel(upload)
        cols = {re.sub(r"\s+","",str(c).strip().lower()): str(c) for c in df.columns}
        col_url = cols.get("url") or cols.get("imageurl") or cols.get("link")
        if not col_url:
            return 0
        urls = df[col_url].astype(str).str.strip()
        return urls.str.startswith(("http://","https://")).sum()
    except Exception:
        return 0

if st.button("Procesar"):
    if not any([f_imgs, f_zip, f_xl]):
        st.warning("Sube imágenes, un ZIP o un Excel/CSV para comenzar.")
        st.stop()
    if not sizes:
        st.error("Selecciona al menos un tamaño destino.")
        st.stop()

    # — Barra de progreso por imagen —
    total_items = 0
    if f_imgs: total_items += len(f_imgs)
    if f_zip:  total_items += count_zip_images(f_zip)
    if f_xl:   total_items += count_excel_rows(f_xl)

    # resetear cursores por si los contadores leyeron el stream
    if f_zip: f_zip.seek(0)
    if f_xl:  f_xl.seek(0)

    done = [0]  # mutable para evitar nonlocal
    prog = st.progress(0.0)

    def tick():
        done[0] += 1
        if total_items > 0:
            prog.progress(min(1.0, done[0] / total_items))

    # 1) sueltas
    if f_imgs:
        for upl in f_imgs:
            try:
                base = safe_name(Path(upl.name).stem)
                img = image_from_bytes(upl.read())
                cut = remove_bg(img, alta_calidad) if usar_quitar_fondo else img
                for sz in sizes:
                    sq = ensure_square_canvas(cut, sz)
                    sub = out_dir / str(sz); sub.mkdir(exist_ok=True, parents=True)
                    p = sub / f"{base}.png"
                    sq.save(p, optimize=True)
                    generated_files.append(p)
                mani_rows.append({"origen":"upload", "sku":"", "url":"", "archivo":base, "estado":"OK"})
            except Exception as e:
                mani_rows.append({"origen":"upload", "sku":"", "url":"", "archivo":upl.name, "estado":f"ERR:{type(e).__name__}"})
            tick()

    # 2) zip
    if f_zip:
        try:
            with zipfile.ZipFile(f_zip) as z:
                z.extractall(work / "zip_in")
            for p in (work / "zip_in").rglob("*"):
                if p.is_file() and is_image_name(p.name):
                    try:
                        img = Image.open(p).convert("RGBA")
                        base = safe_name(p.stem)
                        cut = remove_bg(img, alta_calidad) if usar_quitar_fondo else img
                        for sz in sizes:
                            sq = ensure_square_canvas(cut, sz)
                            sub = out_dir / str(sz); sub.mkdir(exist_ok=True, parents=True)
                            outp = sub / f"{base}.png"
                            sq.save(outp, optimize=True)
                            generated_files.append(outp)
                        mani_rows.append({"origen":"zip", "sku":"", "url":"", "archivo":base, "estado":"OK"})
                    except Exception as e:
                        mani_rows.append({"origen":"zip", "sku":"", "url":"", "archivo":p.name, "estado":f"ERR:{type(e).__name__}"})
                    tick()
        except Exception as e:
            st.error(f"ZIP inválido: {e}")

    # 3) excel/csv
    if f_xl:
        try:
            if f_xl.name.lower().endswith(".csv"):
                df = pd.read_csv(f_xl)
            else:
                df = pd.read_excel(f_xl)

            cols = {re.sub(r"\s+","",str(c).strip().lower()): str(c) for c in df.columns}
            col_url = cols.get("url") or cols.get("imageurl") or cols.get("link")
            col_sku = cols.get("sku")
            if not col_url:
                raise RuntimeError("El archivo debe tener columna 'url' (o 'imageurl'/'link').")

            sess = requests.Session()
            for i, row in df.iterrows():
                sku = str(row[col_sku]).strip() if col_sku else ""
                url = str(row[col_url]).strip()
                if forzar_2000:
                    url = re.sub(r"/img/\d{2,5}/", "/img/2000/", url)

                if not url.lower().startsWith(("http://","https://")):
                    mani_rows.append({"origen":"excel", "sku":sku, "url":url, "archivo":"", "estado":"URL_INVALIDA"})
                    tick(); continue

                img = download_image(url, sess)
                if img is None:
                    mani_rows.append({"origen":"excel", "sku":sku, "url":url, "archivo":"", "estado":"NOT_FOUND"})
                    tick(); continue

                base = safe_name(sku) if sku else f"img_{i+1}"
                try:
                    cut = remove_bg(img, alta_calidad) if usar_quitar_fondo else img
                    for sz in sizes:
                        sq = ensure_square_canvas(cut, sz)
                        sub = out_dir / str(sz); sub.mkdir(exist_ok=True, parents=True)
                        outp = sub / f"{base}.png"
                        sq.save(outp, optimize=True)
                        generated_files.append(outp)
                    mani_rows.append({"origen":"excel", "sku":sku, "url":url, "archivo":base, "estado":"OK"})
                except Exception as e:
                    mani_rows.append({"origen":"excel", "sku":sku, "url":url, "archivo":base, "estado":f"ERR:{type(e).__name__}"})
                tick()
        except Exception as e:
            st.error(f"Error leyendo Excel/CSV: {e}")

    st.success("Proceso finalizado ✅")

    # ===== Manifiesto y descargas =====
    mani = pd.DataFrame(mani_rows)
    st.subheader("Resumen")
    if not mani.empty:
        st.dataframe(mani["estado"].value_counts().rename_axis("estado").reset_index(name="count"))
    else:
        st.write("No hubo entradas procesadas.")

    # manifiesto.xlsx
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        mani.to_excel(w, index=False, sheet_name="manifiesto")
    st.download_button("⬇️ Descargar manifiesto.xlsx",
                       buf.getvalue(),
                       file_name="manifiesto.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    total_files = len(generated_files)
    if total_files == 0:
        st.info("No se generaron imágenes.")
    elif total_files <= 20:
        st.subheader("Descargas individuales (PNG)")
        for p in sorted(generated_files):
            with open(p, "rb") as f:
                st.download_button(f"⬇️ {p.parent.name}/{p.name}", f.read(), file_name=p.name, mime="image/png")
    else:
        st.subheader("Muchas imágenes generadas")
        st.info(f"Se generaron {total_files} archivos. Descarga como ZIP:")
        zbytes = make_zip(out_dir)
        st.download_button("⬇️ Descargar salida.zip", zbytes, "salida.zip", "application/zip")

st.caption("• Salida siempre en PNG (transparente). Si rembg no está instalado, se omite el recorte de fondo.")
if not REMBG_OK:
    st.warning("⚠️ rembg no está disponible: instala con `pip install rembg onnxruntime`.")
