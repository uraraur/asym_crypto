import random
import numpy as np  
import math
from sympy.ntheory import legendre_symbol 

k = 20 #miller rabin moment
l = 256 #довжина p ta q 

def gcd(a, b): 
    r_0, r_1 = a, b 
    u_0, u_1 = 1, 0
    v_0, v_1 = 0, 1

    while r_1 != 0:
        q = r_0 // r_1
        r_0, r_1 = r_1, r_0 - q*r_1
        u_0, u_1 = u_1, u_0 - q*u_1
        v_0, v_1 = v_1, v_0 - q*v_1

    return r_0, u_0, v_0

#Генератор Лемера як генератор випадкових чисел, допоміжні ф-ії------------------------------------------------------------------------------------------

def bits_to_dec(a):
    r = 0
    for i in range(len(a)):
        r += a[len(a) - 1 - i] * 2 ** i
    return r

def l20_gen(t):
    seq = [0] * 8 * t
    for i in range(20):
        seq[i] = np.random.randint(2)
    for j in range(20, 8 * t):
        seq[j] = seq[j - 3] ^ seq[j - 5] ^ seq[j - 9] ^ seq[j - 20]
    return bits_to_dec(seq)

#Перевірка на простоту, генерація випадкових простих чисел------------------------------------------------------------------------------------------

prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def probni_dilenya(n): # n = a_t * 10^t + a_(t-1) * 10^(t-1) + ... + a_1 * 10 + a0 * 1
    n_string = str(n)
    a = []
    for i in n_string[::-1]:
        a.append(int(i))
    
    for m in prime:
        r = np.zeros(len(a))
        r[0] = 1
        for i in range(len(a) - 1):
            r[i + 1] = (r[i] * 10) % m    
        ar = [a * b for a, b in zip(a, r)]
        s = np.sum(ar)
        if s % m == 0:
            return m 
    return 1

def pseudo_prime(x, p, d, s):
    if pow(x, d, p) == 1 or pow(x, d, p) == -1 % p:
        return 1
    x_r = pow(x, 2 * d, p) 
    for i in range(1, s):
        if x_r == -1 % p:
            return 1
        if x_r == 1:
            return 0
        x_r = x_r ** 2 % p
    return 0

def miller_rabin_primality(p, k):
    j = 0
    while j < k:
        if p % 2 == 0:
            return 0
        n = p - 1
        s = 0
        while n % 2 == 0:
            n = n // 2
            s = s + 1
        x = random.randint(2, p - 1)
        if math.gcd(x, p) > 1:
            return 0
        if pseudo_prime(x, p, n, s) == 0:
            return 0
        j = j + 1
    return 1

def if_prime(p): 
    if probni_dilenya(p) == 1:
        if miller_rabin_primality(p, k) == 1:
            return 1

def generate_prime(n):
    r = 0
    while if_prime(r) != 1:
        r = l20_gen(n)
    return r

#Генерація чисел Блюма і обчислення квадратних коренів за модулями Блюма------------------------------------------------------------------------------------------

def generate_blum(n):
    p = generate_prime(n)
    if (p - 3) % 4 != 0:
        p = generate_prime(n)
    return p

def sq_blum(y, p, q):
    s1 = pow(y, (p + 1) / 4, p)
    s2 = pow(y, (q + 1) / 4, q)
    d, u, v = gcd(p, q)
    return [(u * p * s2 + v * q * s1) % (p*q), (u * p * s2 - v * q * s1) % (p*q), (-u * p * s2 + v * q * s1) % (p*q), (-u * p * s2 - v * q * s1) % (p*q)]

#--------------------------------------------------------------------------------------------------------------
def reformat(n, m):
    lenght = ceil(len(bin(n)) / 8)
    if  ceil(len(bin(m)) / 8) < lenght - 10:
        r = random.randint(1, 2**64)
        return 255 * 2**(8 * (lenght - 2)) + m * 2**64 + r
    return "too long m"

def deformat(n, m):
    

class User:
    __p = generate_blum(l)
    __q = generate_blum(l)
    n = __p * __q

    def encrypt(m, n):
        y = pow(m, 2, n)
        c1 = m % 2 
        c2 = int(jacobi_symbol(m, n) == 1)
        return (y, c1, c2)

    def decrypt(self, y, c1, c2):
        sol = sq_blum(y, self.__p, self.__q)
        for s in sol: 
            s_c1 = s % 2 
            s_c2 = int(jacobi_symbol(s, n) == 1)
            if s_c1 == c1 & s_c2 == c2:
                return s



    
        