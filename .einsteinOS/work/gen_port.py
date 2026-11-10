import re,glob,os,sys
os.chdir(os.path.dirname(os.path.abspath(__file__))+'/../..')
files=[f for f in glob.glob('include/**/*',recursive=True)+glob.glob('src/**/*',recursive=True) if os.path.isfile(f)]
txt={f:open(f,errors='ignore').read() for f in files}
def has(f,n): return re.search(r'(?<![A-Za-z0-9_])'+re.escape(n)+r'(?![A-Za-z0-9_])',txt[f]) is not None
rows=[];bad=[]
for l in open('.einsteinOS/work/req.txt'):
    rf,ns=l.strip().split(': ')
    names=ns.split()
    if rf.startswith('src/'):
        tgt=rf   # the new file lives at the reference path
        cands=[tgt] if tgt in txt else []
        if rf=='src/core/native_dense.cpp': cands.append('include/baloon/u11_native_dense.hpp')
    else:
        base=[n.split('::')[-1].lstrip('~') for n in names]
        left=set(base); cands=[]
        while left:
            best=max(files,key=lambda x:(sum(1 for n in left if has(x,n)), x.startswith('include/baloon/')))
            cov={n for n in left if has(best,n)}
            if not cov: break
            cands.append(best); left-=cov
    for n in names:
        base=n.split('::')[-1].lstrip('~')
        pick=None
        for c in cands:
            if n in txt[c] and re.search(r'(?<![A-Za-z0-9_])'+re.escape(n)+r'(?![A-Za-z0-9_])',txt[c]): pick=(c,n);break
        if not pick:
            for c in cands:
                if has(c,base): pick=(c,base);break
        if not pick: bad.append((rf,n)); continue
        ov={('include/ref/kernels/cpu/iq_avx512.hpp','iq512_gu_rows'):'include/baloon/kern_cpu_iq.hpp',
            ('include/ref/kernels/cpu/iq_avx512.hpp','iq512_rows'):'include/baloon/kern_cpu_iq.hpp',
            ('include/ref/kernels/cpu/iq_avx512.hpp','iq512_supported'):'include/baloon/kern_cpu_iq.hpp',
            ('src/core/layout.cpp','Want1'):'include/baloon/rt_layout.hpp',
            ('src/core/layout.cpp','Want2'):'include/baloon/rt_layout.hpp',
            ('src/core/native_dense.cpp','Pending'):'include/baloon/u11_native_dense.hpp',
            ('src/kernels/ngram.cpp','g_ple_prefetch'):'include/baloon/ple_ngram_kernels.hpp'}
        if (rf,n) in ov: pick=(ov[(rf,n)],n)
        rows.append((rf,n,)+pick)
for b in bad: print('UNMAPPED',b)
with open('.einsteinOS/port/u0063babba8.tsv','w') as o:
    for r in rows: o.write('\t'.join(r)+'\n')
print(len(rows),'rows;',len(bad),'unmapped')
