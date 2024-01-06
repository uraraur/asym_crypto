import random
import numpy as np  
import math
from sympy.ntheory import jacobi_symbol
import requests

k = 10 #miller rabin moment
l = 256 #довжина p ta q 

#------------------------------------------------------------------------------------------

def str_to_bin(str):
    res = ""
    res = ''.join(format(ord(i), '08b') for i in str)
    return int(res, 2)

#------------------------------------------------------------------------------------------

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
    seq = [0] * t
    for i in range(20):
        seq[i] = np.random.randint(2)
    for j in range(20, t):
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
    while (p - 3) % 4 != 0:
        p = generate_prime(n)
    return p

def sq_blum(y, p, q):
    s1 = pow(y, (p + 1) // 4, p)
    s2 = pow(y, (q + 1) // 4, q)
    d, u, v = gcd(p, q)
    return [(u * p * s2 + v * q * s1) % (p*q), (u * p * s2 - v * q * s1) % (p*q), (-u * p * s2 + v * q * s1) % (p*q), (-u * p * s2 - v * q * s1) % (p*q)]

#--------------------------------------------------------------------------------------------------------------

def formate(n, m):
    return 255 * 2**(2 * l - 16) + m * 2**64 + random.randint(1, 2**64)

def unformate(n, x):
    return (x % (2**(2 * l - 16))) // (2**64)

class User:
    def __init__(self, l, b = 0):
        self.p = generate_blum(l)
        self.q = generate_blum(l)
        self.n = self.p * self.q
        self.b = random.randint(1, self.n)

    def encrypt(self, m, n):
        x = formate(n, m)
        y = pow(x, 2, n)
        c1 = x % 2 
        c2 = int(jacobi_symbol(x, n) == 1)
        return (y, c1, c2)

    def extend_encrypt(self, m, n, b):
        x = formate(n, m)
        y = (x * (x + b)) % n 
        c1 = ((x + b * pow(2, -1, n)) % n) % 2
        c2 = int(jacobi_symbol(x + b * pow(2, -1, n), n) == 1)
        return (y, c1, c2)

    def decrypt(self, y, c1, c2):
        sol = sq_blum(y, self.p, self.q)
        for s in sol: 
            s_c1 = s % 2 
            s_c2 = int(jacobi_symbol(s, self.n) == 1)
            if s_c1 == c1 & s_c2 == c2:
                return unformate(self.n, s)

    def extend_decrypt(self, y, c1, c2):
        sol = sq_blum((y + pow(self.b, 2, self.n) * pow(4, -1, self.n)), self.p, self.q)
        for s in sol: 
            s_c1 = s % 2 
            s_c2 = int(jacobi_symbol(s, self.n) == 1)
            if (s_c1 == c1 and s_c2 == c2):
                s = (s - self.b * pow(2, -1, self.n)) % self.n 
                return unformate(self.n, s)
    
    def sign(self, m):
        while True:
            x = formate(self.n, m)
            if (jacobi_symbol(x, self.p) == 1 and jacobi_symbol(x, self.q) == 1) :
                break
        return sq_blum(x, self.p, self.q)[random.randint(0, 3)]
        
    def verify(self, s, m, n):
        tx = pow(s, 2, n)
        if unformate(n, tx) != m:
            print(":(")
        return "Success"
    
#--------------------------------------------------------------------------------------------

url = 'http://asymcryptwebservice.appspot.com/rabin/'

s = requests.Session()
res = s.get(f"{url}serverKey?keySize={2 * l}")
s_n = int(res.json()["modulus"], 16)
s_b = int(res.json()["b"], 16)

A = User(l)

sent = "Hello!"
m = str_to_bin(sent)
my_encryp = A.extend_encrypt(m, s_n, s_b)
decryption = s.get(f"{url}decrypt?cipherText={hex(my_encryp[0])[2:]}&expectedType=BYTES&parity={my_encryp[1]}&jacobiSymbol={my_encryp[2]}")
decryption = decryption.json()["message"]
if(int(decryption, 16) == m):
    print("encryption is Successful")

encrypt = s.get(f"{url}encrypt?modulus={hex(A.n)[2:]}&b={hex(A.b)[2:]}&message={hex(m)[2:]}&type=BYTES")
s_y = int(encrypt.json()["cipherText"], 16)
s_c1 = encrypt.json()["parity"]
s_c2 = encrypt.json()["jacobiSymbol"]
my_decryp = A.extend_decrypt(s_y, s_c1, s_c2)
if(my_decryp == m):
    print("decryption is Successful")

sign = s.get(f"{url}sign?message={hex(m)[2:]}&type=BYTES")
sign = int(sign.json()["signature"], 16)
print(f"signature is {A.verify(sign, m, s_n)}ful")

my_sign = A.sign(m)
ver = s.get(f"{url}verify?message={hex(m)[2:]}&type=BYTES&signature={hex(my_sign)[2:]}&modulus={hex(A.n)[2:]}")
print(ver.json())

#----------------------------------------ATtACK-----------------------------------------------------------

session = requests.Session()
server_atacker = session.get("http://asymcryptwebservice.appspot.com/znp/serverKey")
serv_n = int(server_atacker.json()["modulus"], 16)

i = 0
while True:
    i = i + 1
    t = random.randint(0, serv_n)
    y = pow(t, 2, serv_n)
    root = session.get(f"http://asymcryptwebservice.appspot.com/znp/challenge?y={hex(y)[2:]}")
    root = int(root.json()["root"], 16)
    if (t != root or t!= -root):
        d = gcd(t + root, serv_n)
        if (d[0] == 1 or d[0] == serv_n):
            continue
        break
print(f"Number or tries: {i}")