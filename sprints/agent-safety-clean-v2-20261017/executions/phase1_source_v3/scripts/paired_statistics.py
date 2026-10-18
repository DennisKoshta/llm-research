"""Paired binary inference: exact McNemar and Newcombe (1998), method 10.

Reference: Statistics in Medicine 17:2635–2650, Section 5, methods 8 and 10.
DOI: 10.1002/(SICI)1097-0258(19981130)17:22<2635::AID-SIM954>3.0.CO;2-C
"""
from math import sqrt
from statistics import NormalDist
from scipy.stats import binomtest


def wilson(k, n, alpha=0.05):
    if n <= 0 or not 0 <= k <= n:
        raise ValueError('Nonempty valid count required')
    z = NormalDist().inv_cdf(1-alpha/2)
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0.0,center-radius), min(1.0,center+radius)


def paired_binary(treatment, control):
    if not treatment or len(treatment)!=len(control):
        raise ValueError('Complete, nonempty paired observations required')
    if any(v not in (0,1) for v in list(treatment)+list(control)):
        raise ValueError('Binary observations required; do not pass averaged paraphrases')
    e = sum(a==1 and b==1 for a,b in zip(treatment,control))
    f = sum(a==1 and b==0 for a,b in zip(treatment,control))
    g = sum(a==0 and b==1 for a,b in zip(treatment,control))
    h = sum(a==0 and b==0 for a,b in zip(treatment,control))
    n = e+f+g+h
    p1, p2 = (e+f)/n, (e+g)/n
    l1,u1 = wilson(e+f,n)
    l2,u2 = wilson(e+g,n)
    numerator = e*h-f*g
    if numerator>0:
        numerator = max(numerator-n/2,0)
    denominator = sqrt((e+f)*(g+h)*(e+g)*(f+h))
    rho = numerator/denominator if denominator else 0.0
    d = (f-g)/n
    lower = d-sqrt(max(0.0,(p1-l1)**2+(u2-p2)**2-2*rho*(p1-l1)*(u2-p2)))
    upper = d+sqrt(max(0.0,(u1-p1)**2+(p2-l2)**2-2*rho*(u1-p1)*(p2-l2)))
    return {'n_pairs':n, 'both_positive':e, 'treatment_only':f, 'control_only':g, 'both_negative':h,
            'difference':d, 'newcombe_paired_95ci':[lower,upper],
            'newcombe_method':10, 'mcnemar_exact_p':float(binomtest(f,f+g,0.5).pvalue) if f+g else 1.0}


def holm(pvalues):
    if any(not 0<=p<=1 for p in pvalues): raise ValueError('Invalid p-value')
    adjusted = [None]*len(pvalues)
    running = 0.0
    for rank,index in enumerate(sorted(range(len(pvalues)),key=lambda i:pvalues[i])):
        running = max(running,min(1.0,(len(pvalues)-rank)*pvalues[index]))
        adjusted[index] = running
    return adjusted


def ci_gate(comparisons):
    if not comparisons: return False
    return all(c['difference']>=0.15 and c['newcombe_paired_95ci'][0]>=0.10 for c in comparisons)
