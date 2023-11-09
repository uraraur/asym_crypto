#Iмовiрнiсний тест Мiллера-Рабiна та пробні ділення

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
    if probni_dilenya(n) == 1:
        if miller_rabin_primality(n) == 1:
            return 1