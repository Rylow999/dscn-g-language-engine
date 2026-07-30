#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dscng_core.py — Core reutilizable del Language Engine.
Principios:
 - Sin numpy/PyTorch, solo stdlib.
 - Formato de métrica canónico: acc_pred, acc_gt, dolor, foco_acc, W_actual.
"""
import math, random, json, re
from collections import defaultdict, Counter

random.seed(0)

# ---------- matematica basica ----------
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def norm(v):
    return math.sqrt(sum(x * x for x in v))

def cos(a, b):
    na = norm(a); nb = norm(b)
    return 0.0 if na < 1e-9 or nb < 1e-9 else dot(a, b) / (na * nb)

def softmax(vec):
    m = max(vec)
    ex = [math.exp(x - m) for x in vec]
    s = sum(ex)
    return [x / s for x in ex] if s > 0 else [1.0 / len(vec)] * len(vec)

# ---------- metricas ----------
class MetricLogger:
    def __init__(self):
        self.rows = []
        self.agg = defaultdict(float)
        self.count = 0
    def log(self, step, **kwargs):
        row = {"step": step}
        for k in ["acc_pred", "acc_gt", "dolor", "foco_acc", "W_actual"]:
            row[k] = kwargs.get(k)
        self.rows.append(row)
        self.agg["acc_pred"] += row["acc_pred"] or 0.0
        self.agg["acc_gt"]  += row["acc_gt"] or 0.0
        self.agg["dolor"]   += row["dolor"] or 0.0
        self.agg["foco_acc"]+= row["foco_acc"] or 0.0
        self.agg["W_actual"]+= row["W_actual"] or 0.0
        self.count += 1
    def summary(self):
        if not self.count:
            return {}
        return {
            "acc_pred_avg": self.agg["acc_pred"] / self.count,
            "acc_gt_avg": self.agg["acc_gt"] / self.count,
            "dolor_avg": self.agg["dolor"] / self.count,
            "foco_acc_avg": self.agg["foco_acc"] / self.count,
            "W_actual_avg": self.agg["W_actual"] / self.count,
            "steps": self.count,
        }
    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"summary": self.summary(), "rows": self.rows}, f, indent=2)

# ---------- transformer minimo ----------
class SimpleTransformer:
    def __init__(self, vocab, D=16, lr=0.05):
        self.D = D
        self.lr = lr
        self.vocab = vocab
        self.emb = {w: [random.gauss(0, 0.1) for _ in range(D)] for w in vocab}
        self.current = None
    def contexto(self, seq):
        ctx = seq[-8:]
        vec = [0.0] * self.D
        valid = 0
        for w in ctx:
            if w in self.emb:
                for d in range(self.D):
                    vec[d] += self.emb[w][d]
                valid += 1
        if valid > 0:
            vec = [x / valid for x in vec]
        self.current = ctx[-1] if ctx else None
        return vec

# ---------- skip-gram ----------
class SkipGram:
    def __init__(self, vocab, D=16, lr=0.05, window=5, neg_samples=5):
        self.D = D
        self.lr = lr
        self.window = window
        self.neg_samples = neg_samples
        self.vocab = vocab
        self.emb = {w: [random.gauss(0, 0.1) for _ in range(D)] for w in vocab}
        self.ctx = {w: [random.gauss(0, 0.1) for _ in range(D)] for w in vocab}
    def fit(self, tokens, epochs=10):
        rng = random.Random(1)
        for ep in range(epochs):
            for i in range(len(tokens)):
                target = tokens[i]
                if target not in self.emb:
                    continue
                start = max(0, i - self.window)
                end = min(len(tokens), i + self.window + 1)
                for j in range(start, end):
                    if j == i:
                        continue
                    context = tokens[j]
                    if context not in self.ctx or context == target:
                        continue
                    neg = rng.sample(self.vocab, min(self.neg_samples, len(self.vocab)))
                    for d in range(self.D):
                        self.emb[target][d] += self.lr * (self.ctx[context][d] - self.emb[target][d])
                    for ns in neg:
                        for d in range(self.D):
                            self.emb[target][d] -= self.lr * self.ctx[ns][d]

# ---------- root / memoria competitiva ----------
class RootMemory:
    def __init__(self, D, lr=0.05, beta_anchor=0.2, beta_repulse=0.05, theta=0.8, beta_mem=0.05):
        self.D = D
        self.lr = lr
        self.beta_anchor = beta_anchor
        self.beta_repulse = beta_repulse
        self.theta = theta
        self.beta_mem = beta_mem
        self.omega = [random.gauss(0, 0.1) for _ in range(D)]
        self.foco = {"A": 0.5, "B": 0.5}
        self.dolor = 0.0
        self.W_actual = 8
        self.last_veredicto = "EMPATE"
        self.last_diver = None
    def enraizar(self, A, B):
        anch_A = sum(wi * xi for wi, xi in zip(self.omega, A))
        anch_B = sum(wi * xi for wi, xi in zip(self.omega, B))
        paso_A = [self.lr * (x - anch_A * wi) for wi, x in zip(self.omega, A)]
        paso_B = [self.lr * (x - anch_B * wi) for wi, x in zip(self.omega, B)]
        for d in range(self.D):
            self.omega[d] += self.beta_anchor * paso_A[d] + self.beta_anchor * paso_B[d]
            delta = paso_B[d] - paso_A[d]
            signo = 1.0 if (self.foco.get("A", 0) >= self.foco.get("B", 0)) else -1.0
            self.omega[d] += self.beta_repulse * signo * delta
        dist_A = sum((a - b) ** 2 for a, b in zip(self.omega, A))
        dist_B = sum((a - b) ** 2 for a, b in zip(self.omega, B))
        if dist_A < dist_B:
            self.last_veredicto = "A"
            self.last_diver = dist_B - dist_A
            self.dolor = max(0.0, dist_B - dist_A)
            self.foco["A"] += self.beta_mem
        else:
            self.last_veredicto = "B"
            self.last_diver = dist_A - dist_B
            self.dolor = max(0.0, dist_A - dist_B)
            self.foco["B"] += self.beta_mem
        total = sum(self.foco.values())
        if total > 0:
            self.foco = {k: v / total for k, v in self.foco.items()}
        return self.last_veredicto
    def contraer_ventana(self, W_base=8, kappa=0.1):
        self.W_actual = int(max(2, W_base / (1.0 + kappa * (self.dolor or 0.0))))
        return self.W_actual

class LinearSenseClassifier:
    def __init__(self, D, lr=0.05):
        self.D = D
        self.lr = lr
        self.w = [random.gauss(0, 0.1) for _ in range(D)]
        self.b = 0.0
    def predict(self, x):
        return 1 if sum(wi * xi for wi, xi in zip(self.w, x)) + self.b > 0 else 0
    def fit(self, X, Y, epochs=10):
        rng = random.Random(2)
        for ep in range(epochs):
            idx = list(range(len(X)))
            rng.shuffle(idx)
            for i in idx:
                yhat = self.predict(X[i])
                err = Y[i] - yhat
                for d in range(self.D):
                    self.w[d] += self.lr * err * X[i][d]
                self.b += self.lr * err

# ---------- corpus sintetico controlado ----------
def build_polysemy_corpus(word="banco", n_per_sense=350, augmentation=True):
    data = {
        "banco": {
            "A_KEYWORDS": ["dinero","pagar","cuenta","ahorro","plata","banquero","interes","cheque","tarjeta","retiro"],
            "B_KEYWORDS": ["rio","agua","pez","orilla","puente","corriente","boga","remo","proa","popa"],
            "templates_A": [
                "fue al banco para dinero y pagar con tarjeta en mano",
                "el banco aprobo el interes sin plazo ni comision",
                "si tienes ahorro en el banco podras usar el cheque sin credito",
                "cerro su cuenta en el banco despues de retirar el saldo",
                "el banco publico ajusto la tasa de interes por la inflacion",
                "acredite el dinero en el banco para evitar el robo",
            ],
            "templates_B": [
                "se tiro al banco del rio para pescar con su red",
                "el bote choco contra el banco de la orilla al remar",
                "amarraron la barca en el banco mientras el agua subia",
                "cerca del banco se pesco una trucha sobre la arena",
                "el puente esta sobre el banco para cruzarlo temprano",
                "bajamos por el banco del rio hasta la playa",
            ],
        }
    }
    POLYSEMY = data.get(word, data["banco"])
    seq = []
    meta = []
    rng = random.Random(0)
    labels_map = {"A": 1, "B": 0}
    for sense_label in ["A", "B"]:
        key = f"{sense_label}_KEYWORDS"
        keywords = POLYSEMY.get(key, POLYSEMY.get("A_KEYWORDS"))
        tpls = POLYSEMY[f"templates_{sense_label}"]
        # estrategia: muestrear templates con reemplazo y agregar keyword aleatoria del sentido
        for _ in range(n_per_sense):
            sentence = random.choice(tpls)
            # variación sistemática: cambiar una palabra por una keyword del sentido
            if augmentation and rng.random() < 0.7 and len(sentence.split()) > 5:
                tokens = sentence.split()
                pos = rng.randint(1, len(tokens)-2)
                tokens[pos] = random.choice(keywords)
                sentence = " ".join(tokens)
            toks = sentence.split()
            seq.extend(toks)
            meta.extend([sense_label if word in toks else "O" for _ in toks])
    return seq, meta, word, labels_map
