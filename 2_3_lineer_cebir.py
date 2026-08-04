# 02_3_lineer_cebir.py — d2l Ch 2.3 Lineer Cebir
# Amaç: dot product = ağırlıklı toplam, matris-vektör çarpımı = katman, normlar = uzaklık
import torch

# --- dot product: çarp VE topla → tek sayı (skaler) ---
x = torch.tensor([1.0, 2, 3])
w = torch.tensor([4.0, 5, 6])
assert torch.dot(x, w).item() == 32          # 1*4 + 2*5 + 3*6 = 32
# kıyas: eleman-bazlı çarpım toplamaz → vektör kalır
assert (x * w).tolist() == [4, 10, 18]
# dot aslında "çarp sonra topla"nın kısayolu:
assert torch.dot(x, w).item() == (x * w).sum().item()

# --- axis toplama: topladığın eksen KAYBOLUR ---
A = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
assert A.sum(axis=0).shape == (4,)           # satır ekseni gitti → 4 kaldı
assert A.sum(axis=1).shape == (3,)           # sütun ekseni gitti → 3 kaldı
assert A.sum(axis=0).tolist() == [15, 18, 21, 24]
assert A.sum(axis=1).tolist() == [10, 26, 42]

# --- bir nöron = ağırlıklı toplam + bias (ev fiyatı örneği) ---
feat = torch.tensor([120.0, 3])              # [alan m², oda sayısı]
weight = torch.tensor([0.9, 15])             # her girdinin önemi
bias = 10.0
fiyat = torch.dot(feat, weight) + bias
assert fiyat.item() == 163.0                 # 120*0.9 + 3*15 + 10

# --- matris-vektör çarpımı: aynı anda ÇOK ağırlıklı toplam (bir katman) ---
# W'nin her satırı bir "nöron"; sonuç her nöronun ayrı ağırlıklı toplamı
W = torch.tensor([[0.9, 15.0],
                  [1.0,  0.0]])
out = torch.mv(W, feat)                       # 2 nöronun çıktısı
assert out.tolist() == [153.0, 120.0]         # [120*0.9+3*15, 120*1+3*0]

# --- normlar sıfırdan: bir vektörün "uzunluğu" (ileride loss'un temeli) ---
v = torch.tensor([3.0, 4.0])
l2 = torch.sqrt((v**2).sum())                # Öklid uzaklığı
l1 = v.abs().sum()                           # mutlak değerler toplamı
assert l2.item() == 5.0                       # √(9+16) = 5
assert l1.item() == 7.0                       # 3 + 4
assert torch.allclose(l2, torch.norm(v))      # kendi L2'miz torch.norm ile aynı

print("hepsi geçti ✓")