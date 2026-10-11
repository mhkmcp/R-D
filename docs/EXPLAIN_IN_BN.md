# Fault Monitor Demo — বাংলায় ব্যবহার নির্দেশিকা (User Manual)

এই নির্দেশিকা `http://127.0.0.1:8000/ui/` পেজটির (শিরোনাম: **"Image classifier with a built-in
fault monitor"**) প্রতিটি অংশ ধাপে ধাপে ব্যাখ্যা করে — কী দেখা যাচ্ছে, কোন বোতাম কী করে, আর
সংখ্যাগুলোর মানে কী।

---

## ০. এক নজরে: এই পেজটা কী?

পেজটিতে দুটো জিনিস একসাথে চলে:

1. **Image classifier** — একটি প্রশিক্ষিত **ResNet-20** মডেল, যা ৩২×৩২ পিক্সেলের ছোট ছবি দেখে বলে
   ছবিটা CIFAR-10-এর ১০টি শ্রেণির কোনটি: airplane, automobile, bird, cat, deer, dog, frog,
   horse, ship, truck।
2. **Fault monitor (D2 detector)** — মডেলের ভেতরের ৬টি জায়গার (`stem`, `stage1`, `stage2`,
   `stage3`, `pooled`, `logits`) activation পর্যবেক্ষণ করে একটি **score** দেয়। মডেলের weight
   যদি নষ্ট হয়ে যায় (যেমন memory-তে bit flip), তাহলে activation-এর ধরন বদলে যায় এবং score
   বেড়ে যায় — monitor তখন ছবিটিকে **"suspicious"** বলে চিহ্নিত করে।

মূল প্রশ্ন যেটা এই demo দেখায়: **মডেলের memory নষ্ট হলে monitor কি সেটা ধরতে পারে?**

গুরুত্বপূর্ণ বৈশিষ্ট্য:

- সব হিসাব **আপনার browser-এর ভেতরেই** হয় (ONNX Runtime Web দিয়ে `model.onnx` চালিয়ে)। Server
  শুধু ফাইল পাঠায়, কোনো গণনা করে না। Training data লাগে না।
- Bit flip করা হয় **মডেলের একটি কপিতে** — disk-এ থাকা আসল ফাইল কখনো বদলায় না।

---

## ১. পেজটা চালু করা

Export করা bundle ফোল্ডার থেকে:

```bash
cd artifacts/deploy/m2_fp32_default_seed0
python run.py --ui                    # http://127.0.0.1:8000/ui/ খুলবে
python run.py --ui --host 0.0.0.0     # একই নেটওয়ার্কের ফোন থেকে খুলতে চাইলে
```

- শুধু Python-এর standard library লাগে; আলাদা package install করতে হয় না।
- বন্ধ করতে terminal-এ **Ctrl+C**।
- `ui/index.html` ফাইলে **double-click করে খুললে কাজ করবে না** — browser `file://` থেকে মডেল লোড
  করতে দেয় না। তখন পেজে লেখা আসবে *"Could not load the model…"*। সবসময় `run.py --ui` দিয়ে
  চালান।

পেজ খোলার পর নিচে status লাইনে পরপর দেখাবে: *Loading model…* → *Scoring the test pictures…* →
**Ready.** "Ready" না আসা পর্যন্ত বোতামগুলো ধূসর (disabled) থাকে।

---

## ২. পেজের গঠন

পেজে চারটি কার্ড আছে:

| অংশ | শিরোনাম | কাজ |
|---|---|---|
| ১ | Pick a picture | ছবি বাছাই করা, মডেলের উত্তর দেখা |
| ২ | What the monitor says about this picture | ওই একটি ছবির জন্য monitor-এর রায় |
| ৩ | Damage the memory | weight-এ bit flip করে মডেল নষ্ট করা |
| ৪ | Scores on 200 test pictures | ২০০টি ছবির উপর classifier ও monitor-এর মোট ফলাফল |

নিচে footer-এ মডেলের নাম (*fidnn ResNet-20 + D2 fault monitor*) এবং একটি সতর্কবার্তা থাকে: এটি
একটি training seed থেকে export করা demo, thesis-এর রিপোর্ট করা সংখ্যা নয়।

---

## ৩. অংশ ১ — Pick a picture (ছবি বাছাই)

### যা করবেন

- **গ্যালারি থেকে:** নিচের ৪০টি ছোট ছবির যেকোনোটিতে click করুন। নির্বাচিত ছবির চারপাশে নীল
  border আসবে। (mouse রাখলে ছবির সঠিক শ্রেণির নাম দেখায়।)
- **নিজের ছবি:** **"Upload or take a photo"** বোতাম চাপুন। কম্পিউটারে ফাইল বাছাই করা যায়, ফোনে
  সরাসরি ক্যামেরা দিয়ে ছবি তোলা যায়। ছবিটা স্বয়ংক্রিয়ভাবে ৩২×৩২ পিক্সেলে ছোট করা হয়।

### যা দেখবেন

- **বাম পাশে** বড় করে দেখানো ছবি (৩২×৩২, তাই ঝাপসা/পিক্সেলেটেড দেখায় — এটা স্বাভাবিক)।
- **Correct answer: …** — গ্যালারির ছবির আসল শ্রেণি। নিজের ছবি হলে লেখা থাকবে *"Your picture"*।
- **বড় লেখা** — মডেলের অনুমান ও তার confidence, যেমন `cat · 87%`।
- **তিনটি bar** — সবচেয়ে সম্ভাব্য তিনটি শ্রেণি ও তাদের probability।

> মনে রাখবেন: মডেল CIFAR-10-এর মতো ছবিতে প্রশিক্ষিত। একদম অন্যরকম ছবি (screenshot, মানুষের মুখ,
> লেখা) দিলে উত্তর ভুল হতে পারে, এবং monitor-ও fault ছাড়াই সেটাকে suspicious বলতে পারে।

---

## ৪. অংশ ২ — Monitor কী বলছে

### রায় (verdict chip)

- 🟢 **Monitor: looks normal** — এই ছবিতে মডেলের ভেতরের আচরণ স্বাভাবিক।
- 🔴 **Monitor: suspicious** — score threshold পার হয়েছে; মডেলের ভেতরে অস্বাভাবিক কিছু দেখা যাচ্ছে।

মডেল নষ্ট করার পর (অংশ ৩) পাশে আরেকটি chip আসে:

- 🟡 **Answer unchanged by the fault** — নষ্ট মডেল আসল মডেলের মতোই উত্তর দিয়েছে।
- 🔴 **Answer changed by the fault** — fault-এর কারণে উত্তর বদলে গেছে।

### Score meter

- নীল bar = এই ছবির **monitor score**। যত বড়, তত অস্বাভাবিক।
- **লাল দাগ** = threshold। নীল bar লাল দাগ পার হলে ছবিটি flagged।
- নিচে লেখা থাকে, যেমন: `Monitor score 3.41 · threshold 18.28 (red line)`। Score অসীম হলে `∞`
  দেখায় (মডেল খুব বেশি নষ্ট হলে activation-এ NaN/অসীম মান আসে)।

### "Allowed false alarms on a healthy model" (α নির্বাচন)

এটা ঠিক করে threshold কত কড়া হবে। প্রশ্নটা: *মডেল সম্পূর্ণ সুস্থ থাকলেও কত শতাংশ ছবিকে ভুল করে
suspicious বলা মেনে নেবেন?*

| বিকল্প | Threshold | মানে |
|---|---|---|
| 5 % of pictures | 6.45 | সংবেদনশীল — বেশি fault ধরে, কিন্তু সুস্থ মডেলেও ~৫% ছবিতে false alarm |
| **1 % of pictures** (ডিফল্ট) | 18.28 | মাঝামাঝি |
| 0.1 % of pictures | 31.06 | কড়া — false alarm খুব কম, কিন্তু ছোট fault মিস হতে পারে |

এই threshold-গুলো আগেই নির্দিষ্ট করা: শুধু পরিষ্কার (`clean_cal`) ছবির score-এর (1−α) quantile
থেকে নেওয়া (SPEC C3)। Fault data কখনো threshold ঠিক করতে ব্যবহৃত হয়নি (C2)। মান বদলালে অংশ ২ ও
অংশ ৪ দুটোই সাথে সাথে আবার হিসাব হয়।

> **কেন একটা ছবি দিয়ে বিচার করা যায় না:** একটা নষ্ট weight **প্রতিটি** ছবিকে প্রভাবিত করে।
> তাই monitor-কে অনেক ছবির উপর বিচার করতে হয় (অংশ ৪)। সুস্থ মডেলেও প্রায় α শতাংশ ছবি flagged
> হবে — সেটা ত্রুটি নয়, নকশা অনুযায়ী।

---

## ৫. অংশ ৩ — Damage the memory (মডেল নষ্ট করা)

এখানে মডেলের weight-এর কপিতে **bit flip** করা হয় — ঠিক যেমন memory-র hardware fault বা
bit-flip attack (যেমন Rowhammer) করে। প্রতিটি weight একটি ৩২-bit float (FP32)।

### তিনটি পছন্দ

**Where — কোথায়:**

| বিকল্প | কোন layer | মোট weight সংখ্যা (প্রায়) |
|---|---|---|
| Early layers | `conv1`, `layer1.*` | ১৪ হাজার |
| Middle layers | `layer2.*` | ৫০ হাজার |
| Late layers (ডিফল্ট) | `layer3.*`, `fc` | ২ লাখ ৩ হাজার |

**Which bit — কোন bit (FP32 গঠন অনুযায়ী):**

| বিকল্প | Bit নম্বর | প্রভাব |
|---|---|---|
| Sign bit | 31 | weight-এর চিহ্ন উল্টে যায় (+ ↔ −); মান একই থাকে |
| Exponent, top — **drastic** | 27–30 | সবচেয়ে ভয়ংকর। বিশেষ করে bit 30 flip করলে ছোট weight (যেমন 0.05) হঠাৎ ~10³⁷-এর মতো বিশাল হয়ে যায় |
| Exponent, low | 23–26 | মান কয়েক গুণ বাড়ে বা কমে |
| Mantissa, high | 16–22 | মান অল্প (কয়েক শতাংশ পর্যন্ত) বদলায় |
| Mantissa, low — **subtle** | 0–15 | প্রায় অদৃশ্য পরিবর্তন; সাধারণত উত্তরে কোনো প্রভাব নেই |

**How many flips — কতগুলো:** 1, 4 বা 16। নির্বাচিত layer-গ্রুপ ও bit-গ্রুপের মধ্য থেকে
এলোমেলোভাবে এতগুলো আলাদা (weight, bit) জোড়া বাছাই করা হয়।

### বোতাম

- **Flip bits** — উপরের তিনটি পছন্দ অনুযায়ী bit flip করে।
- **Surprise me** — এলোমেলো layer-গ্রুপ, দুটি সবচেয়ে ভয়ংকর bit-গ্রুপের (Sign বা Exponent top)
  একটি, এবং ৪টি flip — নিজে নিজে বেছে নিয়ে প্রয়োগ করে। দ্রুত একটা "বড় fault" দেখতে চাইলে এটা
  চাপুন।
- **Reset model** — সব flip মুছে আসল মডেলে ফিরে যায়।
- ডান পাশের chip: 🟢 **Model intact** অথবা 🔴 **N bit(s) flipped**।

> **Flip জমা হতে থাকে।** "Flip bits" কয়েকবার চাপলে আগের flip-গুলোর সাথে নতুনগুলো যোগ হয়।
> একই (weight, index, bit) দুবার flip হয় না। নতুন করে শুরু করতে **Reset model** চাপুন।

প্রতিটি flip-এর পর পেজ মডেলটি নতুন করে তৈরি করে ২০০টি ছবি আবার চালায় (status-এ *Rebuilding the
model and scoring the test pictures…*), তারপর লেখে *Corrupted model is running.*

### Flip-এর তালিকা

বোতামের নিচে প্রতিটি flip-এর বিবরণ দেখায়, যেমন:

```
layer3.1.conv2.weight[1234] bit 30: 0.04512 → 1.535e+37
```

অর্থ: `layer3.1.conv2.weight` tensor-এর ১২৩৪ নম্বর মানের bit 30 উল্টানো হয়েছে; মান 0.04512 থেকে
হয়ে গেছে 1.535×10³⁷।

### Advanced: flip one exact bit

নির্দিষ্ট একটি bit নিজে বেছে flip করতে "Advanced" খুলুন:

- **Layer** — ২০টি weight tensor-এর তালিকা (নাম · গ্রুপ · কতগুলো সংখ্যা আছে)।
- **Weight index** — ওই tensor-এর কোন মান (0 থেকে মোট সংখ্যা−1)।
- **Bit** — 0 (সবচেয়ে ছোট) থেকে 31 (sign)। ডিফল্ট 30।
- **Flip this bit** চাপুন। ভুল index বা bit দিলে পাশে লাল লেখায় সঠিক সীমা দেখাবে।

---

## ৬. অংশ ৪ — Scores on 200 test pictures (মোট ফলাফল)

২০০টি test ছবি (প্রতি শ্রেণিতে ২০টি) দিয়ে হিসাব। মডেল বা α বদলালেই সব আবার গণনা হয়।

### ৬.১ Classifier: is the answer right?

মডেল কত ভালো শ্রেণি চিনছে:

| টাইল | মানে |
|---|---|
| Accuracy | কত শতাংশ ছবির উত্তর সঠিক |
| Precision (macro) | মডেল কোনো শ্রেণি বললে কত শতাংশ সময় সেটা আসলেই সেই শ্রেণি (১০ শ্রেণির গড়) |
| Recall (macro) | প্রতিটি শ্রেণির ছবির কত শতাংশ মডেল ঠিক চিনেছে (গড়) |
| F1-score (macro) | Precision ও Recall-এর সমান গুরুত্বে মিলিত মান |
| F2-score (macro) | Recall-কে বেশি গুরুত্ব দিয়ে মিলিত মান |

মডেল নষ্ট থাকলে প্রতিটি টাইলের নিচে ছোট করে আসল মডেলের মান দেখায় (`original 91.0%`), যাতে তুলনা
করা যায়। **"Per-class precision, recall and F1"** খুললে প্রতিটি শ্রেণির আলাদা টেবিল পাবেন;
আসল মডেলের চেয়ে কমে যাওয়া মানগুলো **লাল** রঙে দেখায়।

### ৬.২ Fault monitor: does it catch the corrupted model?

**মডেল অক্ষত থাকলে:** ধরার মতো কোনো fault নেই, তাই শুধু **False-alarm rate** মাপা যায়। Recall,
Precision, F1, F2 ফাঁকা (`—`) থাকে, লেখা থাকে *"Flip some bits in section 3 to measure
recall."*

**মডেল নষ্ট করার পর** — হিসাবের পদ্ধতি: প্রতিটি ছবি দুবার চালানো হয়:

- একবার **আসল মডেলে** → *negative* (flag হওয়া উচিত নয়)
- একবার **নষ্ট মডেলে** → *positive* (flag হওয়া উচিত)

#### Confusion matrix

|  | flagged | not flagged |
|---|---|---|
| **corrupted** (নষ্ট মডেল) | 🟢 **TP** — নষ্ট, ধরা পড়েছে | 🔴 **FN** — নষ্ট, কিন্তু মিস হয়েছে |
| **original** (আসল মডেল) | 🔴 **FP** — সুস্থ, তবু ভুল alarm | 🟢 **TN** — সুস্থ, alarm নেই |

#### টাইলগুলো

| টাইল | সূত্র | মানে |
|---|---|---|
| **Recall (detection rate)** | TP / (TP+FN) | নষ্ট মডেলের কত শতাংশ ছবিতে monitor fault ধরেছে — **সবচেয়ে গুরুত্বপূর্ণ** |
| Precision | TP / (TP+FP) | monitor-এর alarm-গুলোর কত শতাংশ আসল fault-এর কারণে |
| F1-score | 2·P·R / (P+R) | Precision ও Recall সমান গুরুত্বে |
| F2-score | 5·P·R / (4·P+R) | Recall-কে ৪ গুণ গুরুত্ব — কারণ fault মিস করা, false alarm-এর চেয়ে খারাপ |
| False-alarm rate | FP / (FP+TN) | সুস্থ মডেলে কত শতাংশ ভুল alarm; নিচে বেছে নেওয়া α দেখায় তুলনার জন্য |
| Accuracy | (TP+TN) / মোট | সব flag/no-flag সিদ্ধান্তের কত শতাংশ ঠিক |
| **Window alarm (3 of 8)** | — | পরপর যেকোনো ৮টি ছবির মধ্যে অন্তত ৩টি flagged হলে 🔴 **ALARM**, নইলে 🟢 **quiet** |

**Window alarm** বাস্তব ব্যবহারের মতো: একটা-দুটো false alarm-এ সতর্ক না হয়ে, ছোট সময়ের মধ্যে
বারবার flag হলে তবেই সংকেত দেয়। মডেল সুস্থ থাকলে এটা *quiet* থাকা উচিত ("should stay quiet"),
নষ্ট হলে *ALARM* হওয়া উচিত ("should fire")।

> **সতর্কতা:** Precision, F1 ও F2 এখানে ৫০/৫০ মিশ্রণ (২০০ সুস্থ + ২০০ নষ্ট) ধরে হিসাব করা।
> বাস্তবে fault খুব বিরল, তাই এই তিনটি মান বাস্তবের চেয়ে ভালো দেখাতে পারে। **Recall ও
> False-alarm rate** এই মিশ্রণের উপর নির্ভর করে না — এ দুটোই নির্ভরযোগ্য মাপকাঠি।

#### "Recall, kept separate by what the fault did to the answer"

নষ্ট মডেলের ছবিগুলোকে দুই ভাগে আলাদা করে recall দেখায়:

| সারি | মানে | Thesis-এর ভাষায় |
|---|---|---|
| **Answer changed by the fault** | fault-এর কারণে উত্তর বদলে গেছে | Track H (SDC/CRASH) — ক্ষতি দৃশ্যমান |
| **Answer unchanged** | উত্তর একই, কিন্তু মডেলের ভেতরে fault আছে | Track S (MASKED/DEGRADED) — নীরব fault |

দুটোকে কখনো মেশানো হয় না। উত্তর না বদলানো fault নিরীহ নয় — শুধু output দেখে সেটা ধরা অসম্ভব।
এখানেই activation-ভিত্তিক monitor-এর আসল মূল্য: output স্বাভাবিক থাকলেও ভেতরের অস্বাভাবিকতা ধরতে
পারা।

#### Reference লাইন

সবার নিচে export-এর সময় ২,০০০ ছবিতে মাপা মান দেখায় — আসল মডেলের accuracy **৯০.৯৫%**, এবং
monitor-এর false-alarm rate: α=5% এ ৪.৬%, α=1% এ ১.৩%, α=0.1% এ ০.১৫%। ২০০ ছবির সংখ্যা এর থেকে
একটু আলাদা হতেই পারে (ছোট নমুনা)।

---

## ৭. ধাপে ধাপে উদাহরণ (প্রথমবার ব্যবহারের জন্য)

1. পেজ খুলে **Ready** আসা পর্যন্ত অপেক্ষা করুন।
2. গ্যালারি থেকে একটা ছবি (যেমন একটা জাহাজ) বেছে নিন → অংশ ১-এ `ship · 9x%` দেখবেন, অংশ ২-এ
   সম্ভবত 🟢 *looks normal*।
3. অংশ ৪-এ দেখুন: Accuracy ~৯০%, False-alarm rate ~১% (α = 1% এর কাছাকাছি), Window alarm 🟢
   *quiet*।
4. অংশ ৩-এ **Where = Late layers**, **Which bit = Mantissa, low — subtle**, **How many = 16** দিয়ে
   **Flip bits** চাপুন। দেখবেন accuracy প্রায় একই, recall খুব কম — এত সূক্ষ্ম fault-এর প্রভাব
   প্রায় শূন্য, তাই ধরাও কঠিন (এবং প্রায়ই ধরার দরকারও পড়ে না)।
5. **Reset model** চাপুন।
6. এবার **Which bit = Exponent, top — drastic**, **How many = 1** দিয়ে **Flip bits** চাপুন।
   সাধারণত accuracy অনেক নেমে যায়, score meter লাল দাগ পার হয়, 🔴 *suspicious*, recall অনেক বেশি,
   Window alarm 🔴 *ALARM*।
7. α বদলে (5 % → 0.1 %) দেখুন recall আর false-alarm rate কীভাবে উল্টো দিকে নড়ে — এটাই
   detection-এর মূল trade-off।
8. **Surprise me** কয়েকবার চেষ্টা করে দেখুন কোন layer-এ fault বেশি ক্ষতিকর আর কোনটা monitor
   সহজে ধরে।

---

## ৮. ছোট শব্দকোষ

| শব্দ | বাংলায় অর্থ |
|---|---|
| Bit flip | memory-র একটি bit 0 থেকে 1 বা 1 থেকে 0 হয়ে যাওয়া |
| Weight | মডেলের শেখা সংখ্যা; এগুলোই memory-তে থাকে এবং নষ্ট হতে পারে |
| Activation | ছবি মডেলের ভেতর দিয়ে যাওয়ার সময় প্রতিটি layer-এর output |
| Tap | যেখানে monitor activation পর্যবেক্ষণ করে (এখানে ৬টি) |
| SVDD | Support Vector Data Description — শুধু সুস্থ data দেখে "স্বাভাবিক" এলাকার সীমা শেখে; সীমার বাইরে গেলে score বাড়ে |
| D2 | সব tap-এর feature একসাথে জুড়ে (fused) একটি SVDD |
| Threshold (τ) | যে score-এর উপরে ছবি flagged হয় |
| α | সুস্থ মডেলে মেনে নেওয়া false alarm-এর হার |
| FP32 | ৩২-bit floating point: ১ sign bit + ৮ exponent bit + ২৩ mantissa bit |
| Macro average | ১০ শ্রেণির মান আলাদা করে হিসাব করে তাদের সাধারণ গড় |

---

## ৯. সীমাবদ্ধতা

- **একটি training seed:** এটা demo; thesis-এর রিপোর্ট করা সংখ্যার জন্য ≥৫ seed ও 95% CI লাগে (C4)।
- **শুধু CIFAR-10-এর মতো ছবি:** false-alarm rate শুধু এমন ছবিতেই প্রযোজ্য। অন্যরকম ছবিতে fault
  ছাড়াই flag হতে পারে।
- **Adversarial defence নয়:** monitor weight-এর নষ্ট হওয়া ধরে; প্রতারণামূলক ছবি (adversarial
  image) থেকে রক্ষা করে না।
- **Browser-এর গতি:** সব গণনা browser-এ (WebAssembly, ১ thread) হয়, তাই প্রতিটি flip-এর পর ২০০টি
  ছবি আবার চালাতে কয়েক সেকেন্ড লাগতে পারে। এখানকার সময় overhead-এর মাপ নয়।
