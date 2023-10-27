import random
import numpy as np  
import scipy
from math import sqrt

alpha = 0.1
n = 800000
r = 10

#Додаткові функції

def cyclic_right_shift(num, k, n):
    return int(((num >> k) | (num << (n - k))) % (2 ** n))

def cyclic_left_shift(num, k, n):
    return int(((num << k) | (num >> (n - k))) % (2 ** n))

def bits_to_bytes(a):
    i = 0
    res = []
    for i in range(0, len(a) - 1, 8):
        x = a[i:i + 8]
        y = 0
        for j in range(len(x)):
            y += 2 ** j * x[j]
        res.append(y)   
    return res

#Тести

def uniformity_test(Y):
    R = {}
    n = len(Y) / 256
    X2 = 0 
    for i in range(256):
        R[i] = 0
    for y in Y:
        R[y] += 1
    for i in range(256):
        X2 += (R[i] - n) ** 2 / n
    X2_l = sqrt(2 * 255) * scipy.stats.norm.ppf(1 - alpha) + 255

    print(f"X2:{X2}")
    print(f"X2_l:{X2_l}")
    if X2 <= X2_l:
        return 1
    return 0

def independence_test(Y):
    R = {}
    R1 = {}
    R2 = {}
    n = round(len(Y) / 2)
    p = 0

    for i in range(256):
        R1[i] = 0
        R2[i] = 0
        for j in range(256):
            R[i, j] = 0

    while 2 * p + 1 <= len(Y):
        R[Y[2*p], Y[2 * p + 1]] += 1
        p = p + 1

    for i in range(256):
        for k in R.keys():
                if k[0] == i:
                    R1[i] += R[k]
        for k in R.keys():
                if k[1] == i:
                    R2[i] += R[k]
    X2 = 0
    for i in range(256):
        for j in range(256):
            if R[i,j] != 0 and R1[i] != 0 and R2[j] != 0 :
                X2 += R[i,j] ** 2 / (R1[i] * R2[j])
    X2 = n * (X2 - 1)
    X2_l = sqrt(2 * 255 ** 2) * scipy.stats.norm.ppf(1 - alpha) + 255 ** 2

    print(f"X2:{X2}")
    print(f"X2_l:{X2_l}")
    if X2 <= X2_l:
        return 1
    return 0    

def homogeneity_test(Y):
    R = {}
    m_len = round(len(Y) / r)
    n = m_len * r

    for i in range(256):
        R[i] = 0
    for y in Y:
        R[y] += 1
    R_interval = np.zeros((r, 256))
    for i in range(r):
        for j in range(m_len):
            num = Y[r * i + j]
            R_interval[i, num] += 1

    X2 = 0
    for i in range(0, r):
        for j in range(256):
            if R[j] != 0:
                X2 += R_interval[i, j] ** 2 / (R[j] * m_len)
    X2 = n * (X2 - 1)
    X2_l = sqrt(2 * 255 * (r - 1)) * scipy.stats.norm.ppf(1 - alpha) +  255 * (r - 1)

    print(f"X2:{X2}")
    print(f"X2_l:{X2_l}")
    if X2 <= X2_l:
        return 1
    return 0      


def test(Y):
    if uniformity_test(Y):
        print("1.Тест на рівноімовірність знаків: :)")
    else: print("1. :(")
    if independence_test(Y):
        print("2.Тест на незалежність знаків: :)")
    else: print("2. :(")
    if homogeneity_test(Y):
        print("3.Тест на однорідність послідовності: :)")
    else: print("3. :(")

#####################################Генератори

#Вбудований:
def integrated_gen(n):
    seq = np.random.randint(256, size=(n))
    return seq

print("\nВбудований генератор:")
test(integrated_gen(n))

#Лемера:
def lemer_gen(state, t):
    m = 2 ** 32
    a = 2 ** 16 + 1
    c = 119 
    seq = np.zeros(t, dtype = int)
    x_i = 0
    x_0 = random.randint(1, 2 ** 8)
    for i in range(t):
        x_i = (a * x_0 + c) % m
        x_0 = x_i
        if state == "low":
            seq[i] = x_i % (2**8)
        elif "high": 
            seq[i] = x_i >> 24
    return seq 

print("\nГенератор Лемера low")
test(lemer_gen("low", n))
print("\nГенератор Лемера high")
test(lemer_gen("high", n))

#L20       
def l20_gen(t):
    seq = [0] * 8 * t
    for i in range(20):
        seq[i] = np.random.randint(2)
    for j in range(20, 8 * t):
        seq[j] = seq[j - 3] ^ seq[j - 5] ^ seq[j - 9] ^ seq[j - 20]
    return bits_to_bytes(seq)

print("\nГенератор L20")
test(l20_gen(n))    

#L89
def l89_gen(t):
    seq = [0] * 8 * t
    for i in range(89):
        seq[i] = np.random.randint(2)
    for j in range(89, 8 * t):
        seq[j] = seq[j - 38] ^ seq[j - 89]
    return bits_to_bytes(seq)

print("\nГенератор L89")
test(l89_gen(n))    

#Geffe
def geffe_gen(t):
    x = np.random.randint(2, size=(11))
    x = np.concatenate((x, np.zeros(8 * t - 11, dtype=int)))
    y = np.random.randint(2, size=(9))
    y = np.concatenate((y, np.zeros(8 * t - 9, dtype=int)))
    for i in range(9, 11):
        y[i] = (y[i - 9] ^ y[i - 8] ^ y[i - 6] ^ y[i - 5])
    s = np.random.randint(2, size=(10))
    s = np.concatenate((s, np.zeros(8 * t - 10, dtype=int)))
    s[10] = (s[0] ^ s[3])
    z = np.zeros(8 * t, dtype = int)
    for i in range(11, 8 * t):
        x[i] = (x[i - 11] ^ x[i - 9])
        y[i] = (y[i - 9] ^ y[i - 8] ^ y[i - 6] ^ y[i - 5])
        s[i] = (s[i - 10] ^ s[i - 7])
    for i in range(8 * t):
        z[i] = s[i] & x[i] ^ (1 ^ s[i]) & y[i]
    return bits_to_bytes(z)

print("\nГенератор Джиффі")
test(geffe_gen(n))    

#Wolfram
def wolfram_gen(t):
    r_0 = random.randint(1, (2 ** 32) - 1)
    seq = np.zeros(8 * t, dtype = int)
    for i in range(8 * t):
        seq[i] = r_0 % 2
        r_1 = (cyclic_left_shift(r_0, 1, 32)  ^ (r_0 | (cyclic_right_shift(r_0, 1, 32)))) 
        r_0 = r_1
    return bits_to_bytes(seq)

print("\nГенератор Вольфрама")
test(wolfram_gen(n))  

#Бібліотекар
def librarian_gen(n, name_f):
    f = open(name_f, 'r', encoding='utf-8')
    text = f.read()
    f.close()
    seq_b = bytes(text, 'utf-8')
    seq = np.zeros(n, dtype= int)
    for i in range(n):
        seq[i] = int(seq_b[i])
    #for s in seq:
    #    print(s)
    return seq

print("\nГенератор Бібліотекар")
test(librarian_gen(n, "At the Edge of Lasglen.txt"))  

#BM
def BM_bits_gen(t):
    seq = np.zeros(8 * t, dtype = int)
    p = 0xCEA42B987C44FA642D80AD9F51F10457690DEF10C83D0BC1BCEE12FC3B6093E3
    a = 0x5B88C41246790891C095E2878880342E88C79974303BD0400B090FE38A688356
    T_0 = random.randint(0, p - 1)
    for i in range(0, 8 * t):
        T_i = T_0
        if T_i < (p-1)/2:
            seq[i] = 1
        elif T_i >= (p-1)/2:
            seq[i] = 0
        T_i = pow(a, T_0, p)
        T_0 = T_i
    return bits_to_bytes(seq)

def BM_bytes_gen(t):
    seq = np.zeros(t, dtype = int)
    p = 0xCEA42B987C44FA642D80AD9F51F10457690DEF10C83D0BC1BCEE12FC3B6093E3
    a = 0x5B88C41246790891C095E2878880342E88C79974303BD0400B090FE38A688356
    T_0 = random.randint(0, p - 1)
    seq[0] = T_0 * 256 // (p - 1)
    for i in range(1, t):
        T_i = pow(a, T_0, p)
        seq[i] = T_i * 256 // (p - 1)
        T_0 = T_i
    return seq

print("\nГенератор Блюма-Мікалі")
test(BM_bytes_gen(n))  

#BBS 
def BBS_bits_gen(t):
    seq = np.zeros(8 * t, dtype = int)
    p = 0xD5BBB96D30086EC484EBA3D7F9CAEB07
    q = 0x425D2B9BFDB25B9CF6C416CC6E37B59C1F
    n = p * q
    r_0 = random.randint(2, n)
    for i in range(8 * t):
        r_i = pow(r_0, 2, n)
        seq[i] = r_i % 2
        r_0 = r_i 
    return bits_to_bytes(seq) 


def BBS_bytes_gen(t):
    seq = np.zeros(t, dtype = int)
    p = 0xD5BBB96D30086EC484EBA3D7F9CAEB07
    q = 0x425D2B9BFDB25B9CF6C416CC6E37B59C1F
    n = p * q
    r_0 = random.randint(2, n)
    for i in range(t):
        r_i = pow(r_0, 2, n)
        seq[i] = r_i % 256
        r_0 = r_i 
    return seq 

print("\nГенератор Блюм-Блюма-Шуба")
test(BBS_bytes_gen(n))  


