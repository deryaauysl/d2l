# 02_1_tensorler.py — d2l Ch 2.1
import torch

# reshape: aynı 12 eleman, farklı biçim
x = torch.arange(12)
X = x.reshape(3, 4)
assert X.shape == (3, 4)
assert X.numel() == 12

# indeksleme: tek indeks 0. eksene (satır) uygulanır
assert X[-1].shape == (4,)          # son satır, 4 elemanlı
assert X[0:2, 1:3].shape == (2, 2)  # alt-blok kırpma (nesne kırpmanın mantığı)

# eleman-bazlı: * matris çarpımı DEĞİL
a = torch.tensor([1.0, 2, 4, 8])
b = torch.tensor([2.0, 2, 2, 2])
assert (a * b).tolist() == [2, 4, 8, 16]

# cat: dim=0 satır ekler, dim=1 sütun ekler
Y = torch.ones(3, 4)
assert torch.cat((X, Y), dim=0).shape == (6, 4)
assert torch.cat((X, Y), dim=1).shape == (3, 8)

p = torch.arange(3).reshape(3, 1)   # (3,1)
q = torch.arange(2).reshape(1, 2)   # (1,2)
assert (p + q).shape == (3, 2)      # broadcasting: (3,1)+(1,2)->(3,2)

# in-place: [:] üstüne yazar, yeni bellek açmaz
Z = torch.zeros_like(Y); before = id(Z)
Z[:] = X + Y
assert id(Z) == before

print("hepsi geçti ✓")