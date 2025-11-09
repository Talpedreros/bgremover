# app.py — PNG transparente, UI corporativa verde, progreso por imagen, sin selecciones por defecto
import io, zipfile, tempfile, re, os
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
import requests

# ===== Ajustes de entorno =====
# Forzar OMP threads para evitar problemas en entornos Cloud
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")

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
  <h1>Catálogo: Quitar fondo y redimensionar</h1>
    <p>Genera imágenes PNG transparentes en múltiples tamaños cuadrados para catálogos online.<br>
    Sube imágenes sueltas, un ZIP o un Excel/CSV con URLs.<br>
    Usa la opción de quitar fondo para obtener PNG transparentes automáticamente.</p>
</div>
""", unsafe_allow_html=True)

# ===== Colocar rembg (seguro) =====
REMBG_OK = False
SESSION = None
try:
    # Importar rembg tras ajustar variables de entorno
    from rembg import remove as rembg_remove, new_session
    # Usar el modelo estable u2net por compatibilidad en Cloud
    try:
        SESSION = new_session("u2net")
    except Exception:
        # fallback a autodetección
        SESSION = new_session()
    REMBG_OK = True
    # Mensaje de debug (se puede quitar en producción)
    st.success("✅ rembg cargado correctamente (modelo: u2net o por defecto)")
except Exception as e:
    REMBG_OK = False
    st.warning(f"⚠️ rembg no está disponible: {e}")

# ===== Utilidades =====
INVALID_RE = re.compile(r"[^\w\-.]+")

def select_output_directory() -> Optional[str]:
    """Selector de carpeta simplificado: mostrar carpeta actual, 'Examinar' y ruta manual.

    El diálogo nativo con tkinter se usa si está disponible (ejecución local). Si no,
    se ofrece el text_input para pegar la ruta manualmente.
    """
    # Intentar importar tkinter para diálogo nativo (solo útil en ejecución local)
    try:
        import tkinter as tk
        from tkinter import filedialog

        def _choose_dir():
            root = tk.Tk()
            root.withdraw()
            try:
                root.attributes('-topmost', True)
            except Exception:
                pass
            path = filedialog.askdirectory()
            root.destroy()
            return path
    except Exception:
        _choose_dir = None

    home = Path.home()

    # UI simplificada: mostrar carpeta actual, botón 'Examinar' y campo manual
    cur = st.session_state.get('output_dir_selected', None)
    disp = cur if cur else "(Usando carpeta temporal por defecto)"
    col1, col2 = st.columns([4,1])
    with col1:
        st.write(f"Carpeta actual: {disp}")
    with col2:
        if st.button("Examinar"):
            if _choose_dir is None:
                st.warning("El diálogo de examen no está disponible en este entorno. Pega la ruta manualmente abajo.")
            else:
                chosen = _choose_dir()
                if chosen:
                    try:
                        Path(chosen).mkdir(parents=True, exist_ok=True)
                        st.session_state['output_dir_selected'] = str(Path(chosen).absolute())
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"No se pudo usar la carpeta seleccionada. Verifica permisos. Detalle: {e}")

    # Fallback: permitir pegar la ruta manualmente
    st.caption("O pega/escribe manualmente una ruta y pulsa 'Usar ruta'.")
    manual_col, btn_col = st.columns([4,1])
    with manual_col:
        manual = st.text_input("Ruta personalizada", value="", placeholder=str(home))
    with btn_col:
        if st.button("Usar ruta manual"):
            if not manual:
                st.warning("Introduce una ruta antes de usarla.")
            else:
                p = Path(manual).expanduser()
                try:
                    p.mkdir(parents=True, exist_ok=True)
                    st.session_state['output_dir_selected'] = str(p.absolute())
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"No se pudo crear la carpeta personalizada. Verifica la ruta y permisos. Detalle: {e}")

    # Devolver la ruta seleccionada (o None si se usa temporal)
    return st.session_state.get('output_dir_selected', None)


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
    """Safely remove background using rembg if available.

    This wrapper catches exceptions (including MemoryError) so the Streamlit
    app doesn't crash on large or problematic images. If rembg is not
    available or an error occurs, the original image is returned and the
    error is logged to the Streamlit UI.
    """
    if not REMBG_OK:
        return img

    try:
        # Protect against extremely large images that may exhaust memory.
        max_side = 1024
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            proc_img = img.resize((nw, nh), Image.LANCZOS)
            st.info(f"Imagen redimensionada a {nw}x{nh}px para procesamiento seguro")
        else:
            proc_img = img
            scale = 1.0

        # Ensure we start with a clean slate
        import gc
        gc.collect()

        if alta_calidad:
            out = rembg_remove(
                proc_img,
                session=SESSION,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_structure_size=10,
                alpha_matting_base_size=1000,
                post_process_mask=True
            )
        else:
            out = rembg_remove(proc_img, session=SESSION)

        # If we downscaled for processing, paste the result onto a canvas sized like original
        if proc_img is not img:
            canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            resized = out.resize((int(round(w*scale)), int(round(h*scale))), Image.LANCZOS)
            canvas.paste(resized, ((w - resized.width) // 2, (h - resized.height) // 2), resized)
            out = canvas

        return out
    except MemoryError as me:
        st.error("Error de memoria al procesar la imagen con rembg. Se devolverá la imagen original.")
        return img
    except Exception as e:
        # Log exception for debugging but don't crash the app
        st.error(f"Error al quitar fondo (rembg): {type(e).__name__}: {e}")
        return img
    finally:
        try:
            import gc
            gc.collect()
        except Exception:
            pass


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

# Límite de tamaño de archivo (10MB para imágenes individuales)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB en bytes

def validate_image_size(uploaded_file):
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"Imagen demasiado grande: {uploaded_file.name} ({uploaded_file.size/1024/1024:.1f}MB). Máximo: 10MB")
        return False
    return True

f_imgs = st.sidebar.file_uploader(
    "Imágenes sueltas (máx. 10MB c/u)", 
    type=["png","jpg","jpeg","webp","bmp","tif","tiff"], 
    accept_multiple_files=True
)

# Validar tamaños de archivos subidos
if f_imgs:
    f_imgs = [f for f in f_imgs if validate_image_size(f)]

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

# ===== Trabajo / salida =====
work = Path(tempfile.mkdtemp(prefix="cat_"))
custom_out_dir = select_output_directory()
out_dir = Path(custom_out_dir) if custom_out_dir else (work / "salida")
out_dir.mkdir(parents=True, exist_ok=True)
mani_rows = []
generated_files: list[Path] = []

if custom_out_dir:
    st.info(f"Las imágenes se guardarán en: {out_dir}")

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
                # Eliminado forzar_2000: ya no se modifica la URL

                if not url.lower().startswith(("http://","https://")):
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
    with st.expander("Ver resumen de procesamiento", expanded=True):
        mani = pd.DataFrame(mani_rows)
        st.subheader("Resumen")
        if not mani.empty:
            st.dataframe(
                mani["estado"].value_counts().rename_axis("estado").reset_index(name="count"),
                use_container_width=True
            )
        else:
            st.write("No hubo entradas procesadas.")

    # manifiesto.xlsx
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        mani.to_excel(w, index=False, sheet_name="manifiesto")
    st.download_button(
        "⬇️ Descargar manifiesto.xlsx",
        buf.getvalue(),
        file_name="manifiesto.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_manifest"
    )

    # Contenedor para descargas
    with st.container():
        total_files = len(generated_files)
        if total_files == 0:
            st.info("No se generaron imágenes.")
        else:
            col1, col2 = st.columns([2,1])
            with col1:
                if total_files <= 20:
                    st.subheader("Descargas individuales (PNG)")
                    for p in sorted(generated_files):
                        with open(p, "rb") as f:
                            # Agregar key único para cada botón basado en la ruta completa
                            button_key = f"download_{p.parent.name}_{p.name}"
                            st.download_button(
                                f"⬇️ {p.parent.name}/{p.name}",
                                f.read(),
                                file_name=p.name,
                                mime="image/png",
                                key=button_key,
                                use_container_width=True
                            )
                else:
                    st.subheader("Descargas agrupadas")
                    st.info(f"Se generaron {total_files} archivos.")
                    zbytes = make_zip(out_dir)
                    st.download_button(
                        "⬇️ Descargar todo como ZIP",
                        zbytes,
                        "salida.zip",
                        "application/zip",
                        key="download_zip",
                        use_container_width=True
                    )

if not REMBG_OK:
    st.warning("⚠️ rembg no está disponible: instala con `pip install rembg onnxruntime`.")
