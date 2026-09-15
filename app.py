"""
Gastronomy AI - Delta Food Detection API v2
Endpoint: POST /analyze  (prev_image + curr_image + weights)
Models:   SegFormer ONNX (seg) + EfficientNet ONNX (clf)
Hosting:  Render - models downloaded from GitHub Releases at cold-start

KEY FIXES vs original app.py:
  1. SegFormer preprocessing: /255 + ImageNet mean/std (confirmed correct)
  2. Class-level delta logic (from notebook), not binary mask subtraction
  3. SEG_TRUST + SEG_CLASS_REMAP from notebook
  4. Classifier: raw 0-255 float32 (confirmed correct from ONNX audit)
  5. Full FOODSEG103 + Indian food class lists
"""

import os, cv2, numpy as np, onnxruntime as ort, urllib.request
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse

# --- CONFIG ---
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
SEGFORMER_ONNX  = os.path.join(BASE_DIR, "segformer_quant.onnx")
CLASSIFIER_ONNX = os.path.join(BASE_DIR, "classifier.onnx")

GITHUB_RELEASE_BASE = "https://github.com/dheerajmalavath/gastronomy-ai-server/releases/download/v1.0"
SEGFORMER_URL  = f"{GITHUB_RELEASE_BASE}/segformer_quant.onnx"
CLASSIFIER_URL = f"{GITHUB_RELEASE_BASE}/classifier.onnx"

# SegFormer: ImageNet normalisation (CONFIRMED correct vs /255-only)
SEG_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
SEG_STD  = np.array([0.229, 0.224, 0.225], np.float32)
SEG_IMG_SIZE  = (512, 512)
SEG_NUM_CLASS = 104
CLF_IMG_SIZE  = (300, 300)
MIN_NEW_BLOB_PX       = 150
PLATE_ARTIFACT_THRESH = 0.45
LOW_CONF_THRESH       = 0.25

# --- FOOD CLASSES ---
FOODSEG103_CLASSES = [
    'background','candy','egg tart','french fries','chocolate','biscuit','popcorn','pudding',
    'ice cream','cheese butter','cake','wine','milkshake','coffee','juice','milk','tea','almond',
    'red beans','cashew','dried cranberries','soy','walnut','peanut','egg','apple','date','apricot',
    'avocado','banana','strawberry','cherry','blueberry','raspberry','mango','olives','peach','lemon',
    'pear','fig','pineapple','grape','kiwi','melon','orange','watermelon','steak','pork',
    'chicken duck','sausage','fried meat','lamb chop','fish fillet','shrimp','hanamaki baozi',
    'wonton dumplings','pasta','noodles','rice','pie','tofu','eggplant','potato','garlic',
    'cauliflower','tomato','kelp','seaweed','spring onion','rape','ginger','okra','lettuce',
    'pumpkin','cucumber','white radish','carrot','asparagus','bamboo shoots','broccoli',
    'celery stick','cilantro mint','snow peas','cabbage','bean sprouts','onion','corn',
    'kidney beans','green beans','french beans','king oyster mushroom','shiitake','enoki mushroom',
    'oyster mushroom','white button mushroom','salad','other ingredients','bread','corn dog','soup','hot dog',
]
while len(FOODSEG103_CLASSES) < SEG_NUM_CLASS:
    FOODSEG103_CLASSES.append(f'cls_{len(FOODSEG103_CLASSES)}')

INDIAN_FOOD_CLASSES = [
    'adhirasam','aloo_gobi','aloo_matar','aloo_methi','aloo_shimla_mirch','aloo_tikki','anarsa',
    'ariselu','bandar_laddu','basundi','bhatura','bhindi_masala','biryani','boondi','butter_chicken',
    'chak_hao_kheer','cham_cham','chana_masala','chapati','chhena_kheeri','chicken_razala',
    'chicken_tikka','chicken_tikka_masala','chikki','daal_baati_churma','daal_puri','dal_makhani',
    'dal_tadka','dharwad_pedha','doodhpak','double_ka_meetha','dum_aloo','gajar_ka_halwa','gavvalu',
    'ghevar','gulab_jamun','imarti','jalebi','kachori','kadai_paneer','kadhi_pakoda','kajjikaya',
    'kakinada_khaja','kalakand','karela_bharta','kofta','kuzhi_paniyaram','lassi','ledikeni',
    'litti_chokha','lyangcha','maach_jhol','makki_di_roti_sarson_da_saag','malapua','misi_roti',
    'misti_doi','modak','mysore_pak','naan','navrattan_korma','palak_paneer','paneer_butter_masala',
    'phirni','pithe','poha','poornalu','pootharekulu','qubani_ka_meetha','rabri','ras_malai',
    'rasgulla','sandesh','shankarpali','sheer_korma','sheera','shrikhand','sohan_halwa','sohan_papdi',
    'sutar_feni','unni_appam',
]

SEG_CLASS_REMAP = {'kelp': 'rice', 'seaweed': 'rice'}
SEG_TRUST = {
    'egg','rice','steak','pork','chicken duck','sausage','fish fillet','shrimp',
    'noodles','pasta','soup','corn','tomato','banana','watermelon','orange','apple',
    'mango','broccoli','carrot','cucumber',
}
CALORIE_MAP = {
    'biryani':1.8,'butter_chicken':2.4,'dal_tadka':1.2,'chapati':3.0,'naan':2.9,
    'aloo_gobi':1.1,'gulab_jamun':3.2,'paneer_butter_masala':2.2,'poha':1.5,
    'dal_makhani':1.5,'palak_paneer':1.8,'chana_masala':1.4,'aloo_matar':1.2,
    'kadai_paneer':2.0,'dum_aloo':1.6,'gajar_ka_halwa':2.5,'jalebi':3.0,
    'rasgulla':1.8,'rice':1.3,'egg':1.6,'soup':0.5,
}
DEFAULT_CAL = 1.6

# --- MODEL DOWNLOAD ---
def download_model(url, dest):
    if not os.path.exists(dest):
        print(f"[Boot] Downloading {os.path.basename(dest)} ...")
        import urllib.request
        # Stream in 8 MB chunks — avoids loading full file into RAM at once
        with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
            chunk = 8 * 1024 * 1024  # 8 MB chunks
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                f.write(data)
        print(f"[Boot] Done: {os.path.getsize(dest)//1024//1024} MB")
    else:
        print(f"[Boot] Found {os.path.basename(dest)} ({os.path.getsize(dest)//1024//1024} MB)")

download_model(SEGFORMER_URL,  SEGFORMER_ONNX)
download_model(CLASSIFIER_URL, CLASSIFIER_ONNX)

# --- LOAD MODELS ---
print("[Boot] Loading ONNX sessions ...")
_providers   = ['CPUExecutionProvider']
seg_session  = ort.InferenceSession(SEGFORMER_ONNX,  providers=_providers)
clf_session  = ort.InferenceSession(CLASSIFIER_ONNX, providers=_providers)
_clf_inp     = clf_session.get_inputs()[0].name
print("[Boot] Both models ready.")

app = FastAPI(title="Gastronomy AI", version="2.0")

# --- HELPERS ---
def decode_image(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image bytes")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def seg_predict(img_rgb: np.ndarray):
    img512 = cv2.resize(img_rgb, SEG_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    arr = (img512.astype(np.float32)/255.0 - SEG_MEAN) / SEG_STD  # ImageNet norm
    inp = arr.transpose(2,0,1)[np.newaxis]
    logits = seg_session.run(None, {"image": inp})[0]
    mask = np.argmax(logits, axis=1)[0].astype(np.uint8)
    return mask, img512

def filter_plate(mask):
    out = mask.copy()
    uid, cnt = np.unique(mask, return_counts=True)
    for u,c in zip(uid,cnt):
        if u > 0 and c/mask.size > PLATE_ARTIFACT_THRESH:
            print(f"  [plate] removed {FOODSEG103_CLASSES[int(u)]} ({c/mask.size*100:.0f}%)")
            out[out==u] = 0
    return out

def clean_seg(mask, min_px=400):
    binary = (mask>0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    n,lbl,stats,_ = cv2.connectedComponentsWithStats(closed,8)
    keep = np.zeros_like(binary)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA] >= min_px: keep[lbl==i]=1
    out = mask.copy(); out[keep==0]=0
    return out

def get_delta_mask(prev, curr):
    new_cls = (set(np.unique(curr).tolist()) - set(np.unique(prev).tolist())) - {0}
    if new_cls:
        print(f"  [delta] new classes: {[FOODSEG103_CLASSES[c] for c in sorted(new_cls)]}")
    new_mask = np.zeros_like(curr)
    for c in new_cls: new_mask[curr==c] = c
    binary = (new_mask>0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    n,lbl,stats,_ = cv2.connectedComponentsWithStats(binary,8)
    out = np.zeros_like(curr)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA] >= MIN_NEW_BLOB_PX:
            out[lbl==i] = curr[lbl==i]
    return out

def classify_crop(img, blob_bin):
    ys,xs = np.where(blob_bin); m=8
    y1=max(0,ys.min()-m); y2=min(img.shape[0],ys.max()+m)
    x1=max(0,xs.min()-m); x2=min(img.shape[1],xs.max()+m)
    crop = img[y1:y2,x1:x2].copy()
    masked = np.full_like(crop,128); masked[blob_bin[y1:y2,x1:x2]] = crop[blob_bin[y1:y2,x1:x2]]
    resized = cv2.resize(masked, CLF_IMG_SIZE).astype(np.float32)  # raw 0-255 confirmed
    probs = clf_session.run(None, {_clf_inp: resized[np.newaxis]})[0][0]
    top = np.argsort(probs)[::-1][:3]
    return [(INDIAN_FOOD_CLASSES[j] if j<len(INDIAN_FOOD_CLASSES) else f'idx_{j}', float(probs[j])) for j in top]

def run_pipeline(prev_rgb, curr_rgb, delta_weight):
    if curr_rgb.shape != prev_rgb.shape:
        curr_rgb = cv2.resize(curr_rgb, (prev_rgb.shape[1], prev_rgb.shape[0]))

    prev_mask_raw, _       = seg_predict(prev_rgb)
    curr_mask_raw, curr512 = seg_predict(curr_rgb)
    prev_mask = clean_seg(filter_plate(prev_mask_raw))
    curr_mask = clean_seg(filter_plate(curr_mask_raw))

    new_mask = get_delta_mask(prev_mask, curr_mask)
    if not np.any(new_mask):
        return {"status":"no_new_food","reason":"No new seg class detected","dish":None,
                "confidence":0.0,"weight_g":round(delta_weight,1),"calories_kcal":0.0}

    new_bin = (new_mask>0).astype(np.uint8)
    n,lbl,stats,_ = cv2.connectedComponentsWithStats(new_bin,8)
    candidates = [(stats[i,cv2.CC_STAT_AREA],i) for i in range(1,n)
                  if stats[i,cv2.CC_STAT_AREA]>=MIN_NEW_BLOB_PX]
    if not candidates:
        return {"status":"no_new_food","reason":"All delta blobs too small","dish":None,
                "confidence":0.0,"weight_g":round(delta_weight,1),"calories_kcal":0.0}

    best_area, best_i = max(candidates)
    blob_bin = (lbl==best_i)
    vals = new_mask[blob_bin]; vals = vals[vals>0]
    if len(vals) == 0:
        return {"status":"no_new_food","reason":"Blob has no class pixels","dish":None,
                "confidence":0.0,"weight_g":round(delta_weight,1),"calories_kcal":0.0}
    uid,cnt = np.unique(vals, return_counts=True)
    seg_raw  = FOODSEG103_CLASSES[int(uid[cnt.argmax()])]
    seg_name = SEG_CLASS_REMAP.get(seg_raw, seg_raw)
    if seg_name != seg_raw: print(f"  [remap] {seg_raw} -> {seg_name}")

    if seg_name in SEG_TRUST:
        dish,confidence,topk,source = seg_name,1.0,[(seg_name,1.0)],"seg"
    else:
        topk = classify_crop(curr512, blob_bin)
        confidence = topk[0][1]
        dish = seg_name if confidence<LOW_CONF_THRESH else topk[0][0]
        source = "seg-lowconf" if confidence<LOW_CONF_THRESH else "clf"

    calories = round(delta_weight * CALORIE_MAP.get(dish, DEFAULT_CAL), 1)
    return {
        "status":"success","dish":dish,"confidence":round(confidence,4),
        "seg_class":seg_name,"seg_raw":seg_raw,"source":source,
        "top3":[{"label":l,"conf":round(c,4)} for l,c in topk],
        "area_px":int(best_area),"weight_g":round(delta_weight,1),"calories_kcal":calories,
    }

# --- ENDPOINTS ---
@app.get("/")
def health():
    return {"status":"online","service":"Gastronomy AI v2","endpoint":"POST /analyze"}

@app.post("/analyze")
async def analyze(
    prev_image:    UploadFile = File(...),
    curr_image:    UploadFile = File(...),
    prev_weight_g: float      = Form(...),
    curr_weight_g: float      = Form(...),
):
    try:
        img_prev = decode_image(await prev_image.read())
        img_curr = decode_image(await curr_image.read())
        delta_w  = max(0.0, curr_weight_g - prev_weight_g)
        if delta_w < 5.0:
            return JSONResponse({"status":"ignored",
                "reason":f"Weight delta {delta_w:.1f}g < 5g threshold",
                "delta_weight_g":round(delta_w,1)})
        return JSONResponse(run_pipeline(img_prev, img_curr, delta_w))
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"status":"error","message":str(e)})


@app.post("/analyze_batch")
async def analyze_batch(request: Request):
    """
    Batch delta endpoint called by ESP32 SEND button.
    Accepts N images (image_0..image_{N-1}) + N weights (weight_0..weight_{N-1}).
    Processes N-1 consecutive deltas and returns all results.
    """
    try:
        form = await request.form()

        # Extract all images and weights dynamically
        images, weights = [], []
        i = 0
        while True:
            img_key = f"image_{i}"
            wt_key  = f"weight_{i}"
            if img_key not in form:
                break
            img_bytes = await form[img_key].read()
            images.append(decode_image(img_bytes))
            weights.append(float(form[wt_key]))
            i += 1

        N = len(images)
        if N < 2:
            return JSONResponse(status_code=400, content={
                "status": "error",
                "message": f"Need at least 2 images, got {N}"
            })

        print(f"[Batch] Processing {N} snaps -> {N-1} deltas")

        results = []
        for step in range(N - 1):
            delta_w = max(0.0, weights[step+1] - weights[step])
            print(f"[Batch] Step {step+1}/{N-1}: delta={delta_w:.1f}g")

            if delta_w < 5.0:
                results.append({
                    "step": step + 1,
                    "status": "ignored",
                    "reason": f"Weight delta {delta_w:.1f}g < 5g",
                    "dish": None,
                    "confidence": 0.0,
                    "weight_g": round(delta_w, 1),
                    "calories_kcal": 0.0,
                })
                continue

            r = run_pipeline(images[step], images[step+1], delta_w)
            r["step"] = step + 1
            results.append(r)

        total_kcal = sum(r.get("calories_kcal", 0) for r in results)
        return JSONResponse({
            "status": "success",
            "count": len(results),
            "total_calories_kcal": round(total_kcal, 1),
            "results": results,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)



