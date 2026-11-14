"""Supplementary Fig. 9: calibrated separation of the 21-configuration catalogue.
Reads evidence/figS_separability.csv only (frozen values); no number of its own."""
import sys, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
get = lambda arm, ses: next(r for r in rows if r['arm'] == arm and r['session'].startswith(ses))
items = [('No lens, catalogue sessions', get('no lens', 'EXP 04'), '#BD7C48'),
         ('No lens, profile sessions', get('no lens', 'EXP 06'), '#BD7C48'),
         ('With lens, catalogue sessions', get('with lens', 'EXP 04'), '#5089AD')]
need = float(rows[0]['needed_sigma'])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.linewidth': 0.6,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(180/25.4, 42/25.4))
ys = [2, 1, 0]
for y, (lab, r, col) in zip(ys, items):
    v = float(r['codebook_d_min_sigma'])
    ax.barh(y, v, left=1, height=0.56, color=col, edgecolor='none', zorder=2)
    xr = v
    if r['ci_lo']:
        lo, hi = float(r['ci_lo']), float(r['ci_hi'])
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt='none', ecolor='#222222', elinewidth=0.9,
                    capsize=2.5, capthick=0.9, zorder=3)
        xr = hi
    ax.text(xr * 1.12, y, f'{v:.1f}', va='center', ha='left', fontsize=7)
ax.axvline(need, color='#222222', lw=0.9, ls=(0, (4, 2)), zorder=4)
ax.text(need * 1.08, 2.62, f'sufficient reference, {need:.1f}', va='center', ha='left', fontsize=7)
ax.set_xscale('log'); ax.set_xlim(1, 2000); ax.set_ylim(-0.6, 2.95)
ax.set_yticks(ys); ax.set_yticklabels([i[0] for i in items])
ax.tick_params(axis='y', length=0); ax.tick_params(axis='x', width=0.6, length=3)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.set_xlabel(r'Minimum normalized separation $d_{\min}$ of the 21-configuration catalogue')
fig.subplots_adjust(left=0.235, right=0.985, top=0.97, bottom=0.27)
fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=300)
print('written', out)
