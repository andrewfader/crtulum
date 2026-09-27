"""Regenerate positive reservoirs from published P22 impulse-response equations."""
import math,json
terms=[]
# Published exponential amplitudes and pole frequencies: Kuhn 2002 eqs. 8-10.
r=[(4,360),(1.75,1600),(2,8000),(2.25,25000),(15,700000),(29,7000000)]
g=[(37,150000),(100,700000),(90,5000000)]
b=[(75,100000),(1000,1100000),(1100,4000000)]
for i in range(6):
    amplitudes=[];rates=[]
    for series in [r,g,b]:
        a,f=series[i] if i<len(series) else (0,1)
        amplitudes.append(a); rates.append(2*math.pi*f)
    terms.append([rates,amplitudes])
# Positive log-Laplace quadrature of (t+alpha)^-beta, retaining superposition.
step=math.log(1e11)/24
for i in range(24):
    rate=1e-4*math.exp((i+.5)*step)
    amps=[0]+[a/math.gamma(beta)*rate**beta*math.exp(-alpha*rate)*step for a,alpha,beta in [(210e-6,5.5e-6,1.1),(190e-6,5e-6,1.11)]]
    terms.append([[rate]*3,amps])
# Normalize by the analytic total energy, not the finite quadrature sum.
totals=[sum(a/(2*math.pi*f) for a,f in series) for series in [r,g,b]]
for j,(a,alpha,beta) in enumerate([(210e-6,5.5e-6,1.1),(190e-6,5e-6,1.11)],1): totals[j]+=a*alpha**(1-beta)/(beta-1)
# The far tail beyond the slowest pole is real. Lump its integrated energy into
# one slower pole; it contributes negligibly over our 1us..1s fit interval.
for rate,amp in terms:
    for c in range(3): amp[c]/=rate[c]*totals[c]
residual=[max(0,1-sum(t[1][c] for t in terms)) for c in range(3)]
terms.append([[1e-5]*3,residual])
with open('src/phosphor.wgsl','w') as f:
    f.write('// Measured P22 response: Kuhn, IEEE S&P 2002, equations 8–10.\n// Positive exponential reservoirs approximate the sulfide power-law tails.\n')
    for name,col in [('PHOS_RATE',0),('PHOS_ENERGY',1)]:
        f.write(f'var<private> {name}: array<vec3<f32>, 31> = array<vec3<f32>, 31>(\n')
        for term in terms:f.write('    vec3<f32>('+', '.join(f'{v:.9e}' for v in term[col])+'),\n')
        f.write(');\n')
with open('docs/phosphor-response.json','w') as f:json.dump({'source':'https://www.cl.cam.ac.uk/~mgk25/ieee02-optical.pdf','terms':[{'rate_per_second':r,'energy_fraction':w} for r,w in terms]},f,indent=2)
# Verify instantaneous response normalized to area against independent exact formulas.
worst=0
for j in range(301):
    t=10**(-6+j/300*6)
    for c,ss in enumerate([r,g,b]):
        val=sum(a*math.exp(-2*math.pi*fr*t) for a,fr in ss)
        if c: a,al,be=[(210e-6,5.5e-6,1.1),(190e-6,5e-6,1.11)][c-1]; val+=a*(t+al)**(-be)
        expected=val/totals[c]
        actual=sum(rates[c]*weights[c]*math.exp(-rates[c]*t) for rates,weights in terms)
        if expected>1e-12:worst=max(worst,abs(actual/expected-1))
print('max relative response error (1us..1s):',worst)

assert worst < 0.002, "phosphor response fit exceeded 0.2%"
