"""Author-requested continuation of the professor's v7 figures, 2026-11-06.

Run from the repository root with Python -B paper/scripts/figure_revision.py.
Frozen v7 inputs stay unchanged. Figures 1/3 adapt the supplied drawing code;
new scalar fits and priors use the existing direct package and EXP06 loaders.
The current main manuscript includes these figures directly.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle, Patch
from matplotlib.transforms import Affine2D, Bbox

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'paper/scripts'))
import figures as local
from direct.recover import (fit_known_pattern, pattern_budget_sweep,
                            relative_profile_rms, tsvd_profile_risk, profile_recovery_budget,
                            truncated_pinv)
from direct.mode_prior import operator, real_operator
from direct.prior import gaussian_statistic_partial_prior
from professor_v7 import style, loaders, source_data
from professor_v7 import fig01_hero as hero
from professor_v7 import fig03_budget as budget
from professor_v7 import fig02_thresholds as thresholds

OUT = ROOT / 'paper/figures/main/current'
DATA = loaders.DATA_DIR
COLOURS = {'nolens': style.C_NOL, 'withlens': style.C_LENS}
PRIOR_COLOURS = {'none': style.C_NOL, 'correct': style.C_LENS, 'wrong': '#9A7B7D'}
PATTERNS = ('sin_p1', 'sin_p2', 'cos_p4')
LABELS = ('λ/6.5 (sin)', 'λ/13 (sin)', 'λ/26 (cos)')
AMPLITUDE_RTOL = 0.04  # Whole-percent ceiling of the empirical 95th-percentile case RMS.
PROFILE_RTOL = 1/np.sqrt(2)  # Operational half-relative-mean-square-error tolerance.

CAPTIONS = {1: 'Two given sources can produce strong but indistinguishable recordings. (a) Unit-norm mirror-related '
    'model sources s₁ and s₂ are the unchanged orthogonal v₁ ± v₈ pair. Signed drives use the '
    'blue–white–coral map and redundant minus marks; the photograph shows the apparatus. (b) Computed '
    'Re[p(s₁) − p(s₂)] at the original selected tone, divided by the common complex noise rms σ. The asinh '
    'compression and ±300σ limits are unchanged. Insets show the total field of s₁. Filled microphones '
    'denote |Δp| ≥ σ at this tone, a display convention. (c) Each curve is the model distribution of a '
    'scalar recording projection for one fixed candidate. The full recording y stacks six microphones and '
    'twenty frequencies. Set yᵢ = Tsᵢ, Δy = y₂ − y₁ and ȳ = (y₁ + y₂)/2. The horizontal coordinate is z = '
    'Re[Δyᴴ(y − ȳ)]/[(σ/√2)||Δy||]: projection onto the difference of the predicted recordings, centred at '
    'their midpoint and divided by the noise standard deviation of that projection. Assumed proper-complex '
    'Gaussian noise gives z|s₁ ∼ N(−d/2,1) and z|s₂ ∼ N(d/2,1), with d = √2||Δy||/σ; every curve therefore '
    'has noise standard deviation 1. Without the lens d = 0.00872 and the means nearly coincide at '
    '±0.00436. With the lens d = 83.6558, so the means are ±41.8279, labelled on the axis as ±42. These are '
    'noise-normalized projection values. Teal solid and magenta dashed curves match the pictured '
    'candidates. Both rows use 2 mm per noise standard deviation; the with-lens axis is visibly broken. '
    'The black dot and stem mark the mean recording of s₁. Percentages below the candidate pictures give '
    'their conditional probabilities for that recording with equal initial weights, rather than average '
    'classification accuracy. Lens probabilities are displayed as bounds >99.9% and <0.1%. The sources, '
    'fields, noise, camera and probabilities are unchanged.',
 5: 'Source knowledge with a flat data-only baseline. Warm orange squares, blue upward triangles and muted '
    'rose downward triangles identify data only, compatible prior and wrong prior; black stars are truth. '
    'Blue and rose arrows run from the actual data-only estimate to the compatible- and wrong-prior estimates. Their numeric labels '
    'give the change in full-source relative profile error in percentage points (pp). Separate insets '
    'enlarge each prior shift in the with-lens panels by the labelled factor in both '
    'physical display directions, using the marked parent window. Each shows the data-only and '
    'corresponding prior estimate, including the neural estimate where present; '
    'truth remains in the main axes. No point is displaced for visibility. Axes show signed strengths of '
    'the physical patterns drawn at their corresponding axes. Both arms retain common basis, priors and '
    'limits within each row. Adjacent source profiles share division by the true profile peak. Percentages are full-source errors, without amplitude '
    'refitting. (a,b) The unchanged model example has two unknown real strengths in directions 3 and 5 of '
    'the fixed with-lens real-stack reference basis; all other directions are fixed to zero. Operators use '
    'their own σ₁ normalization at 24 dB, with prescribed standardized displacement (0.4,−0.55). Data only '
    'is unregularized least squares; zero-mean priors have unit marginal variances and correlations ±0.9. '
    'Shading is the data-only Gaussian likelihood contour at Mahalanobis distance 2 for exactly known operators. Hollow purple '
    'circles and dotted profiles add independently trained 2–32–2 ReLU decoders for the same two '
    'Gaussian priors and measurement laws (fixed seed 101; three training seeds checked). They learn '
    'true coefficients from independent source/noise draws, not posterior labels or the displayed truth. '
    'Printed errors belong to the analytic profiles; neural outputs and validation are in Source Data. (c,d) The '
    'unchanged recorded gap pair at speakers 1 and 3 uses all five repeats; all eight real source '
    'drive coefficients remain free. Bias and full error covariance are fitted on the other seventeen class means, '
    'holding out every target record. Data only is bias-corrected least squares. Zero-mean soft priors '
    'place variance 1 on positions (1,3), or the wrong (1,2), and 0.01 elsewhere. Full covariance is '
    'transformed into the fixed no-lens reference basis, displaying directions 5 and 7. Shading is a Mahalanobis-distance-2 '
    'empirical working-error contour, not a validated confidence region. These selected examples do not '
    'establish general risk improvement.'}

# The adopted reconstruction test scores the entire complex profile in all three figures.
CAPTIONS.update({
 2: 'Detection, scalar amplitude estimation and free-source reconstruction in the model. '
    '(a) No-lens microphone powers and background bands at 64 dB. (b) Scalar RMS amplitude '
    'errors for both arms, with the unchanged 4% tolerance. (c) Frozen first noisy TSVD '
    'profiles in both arms; curves show real parts and stems imaginary parts, divided by '
    'the true pattern peak only for display. The gauge and printed values use the exact '
    'RMS relative profile error E_prof over the assumed proper-complex white noise, '
    'including all eight complex source components without amplitude or phase refitting. '
    'The common reconstruction criterion is E_prof <= 1/sqrt(2), equivalent to '
    'relative mean-square error at most one half. No-lens/with-lens floor ranks are three/eight. '
    'The three RMS errors are 30.6/93.8/97.6% without and 2.4% each with the lens.',
 3: 'Minimum required budgets for detection, known-pattern amplitude estimation and '
    'full-profile reconstruction on calibrated acoustic operators. (a) Acquired driven '
    'powers relative to background. (b) Total-noise detection. (c) Scalar RMS amplitude '
    'error at most 4%. (d) Complex floor-rank TSVD with RMS relative profile error '
    'E_prof <= 1/sqrt(2), the same metric and tolerance as Figures 2 and 4. The exact '
    'squared error is the discarded source-norm fraction plus noise variance summed '
    'over all retained inverse directions. Marks give the first 0.1-dB grid budget '
    'after which the criterion stays satisfied. The operator uses baseline-standard-'
    'deviation row weights and assumed independent proper-complex white noise. '
    'Panels b-d are calculations on measured operators, not acquired noise sweeps. '
    'The detection/amplitude panels and their source patterns are unchanged.',
 4: 'Measured detection, scalar amplitude estimation and free-source reconstruction. '
    '(a) Acquired powers and background bands. (b) Five observed scalar amplitude '
    'errors, their medians and the unchanged conditional epsilon/g limits under the '
    'assumed complete-error bound 0.03*sigma1*norm(a0), with the 4% scalar tolerance. '
    'The wide 0-10% axes show every observed error and the 4% target; aligned narrow '
    '0-65% axes separately show the conditional limits. The two scales are labelled '
    'independently; no measurement or limit is omitted. '
    '(c) Unchanged first-record complex TSVD profiles at hard rank five and the first '
    'individual-baseline weights, without phase or amplitude refitting. Printed '
    'E_prof values are RMS relative profile errors across all five separate recordings, '
    'including their imaginary components. The same E_prof <= 1/sqrt(2) criterion '
    'as Figures 2 and 3 passes zero of three no-lens cases and all three with the lens. '
    'For the finest pattern, the five-record RMS error changes from 0.94 to 0.63. '
    'The five-record score is not the error of the one displayed recording.'})


def save(fig, number):
    # The hero's photograph and rasterized fields need the original 600-dpi
    # PDF export; Matplotlib's default 100 dpi visibly downsamples the photo.
    paths = style.save(fig, OUT, f'Fig{number}',
                       pdf_dpi=hero.PDF_DPI if number==1 else None)
    print('wrote', *(p.name for p in paths), flush=True)


def letter(fig, x, y, value):
    style.tx_mm(fig, x, y, f'({value})', weight='bold', fontsize=8)


def box(ax):
    style.box(ax)
    ax.tick_params(labelsize=7, pad=2)


def vector_thumbnail(fig,x,y,width,height,values):
    """Eight acquired drive components; peak normalization is for the glyph only."""
    values=np.asarray(values,float)
    v=values/np.max(abs(values));speakers=np.arange(len(v))
    ax=style.ax_mm(fig,x,y,width,height)
    ax.set_label('source_vector')
    ax.axhline(0,color=style.RULE,lw=.5)
    ax.vlines(speakers,0,v,color=style.C_LENS,lw=.45,alpha=.65)
    ax.plot(speakers,v,'o-',color=style.C_LENS,lw=.55,ms=1.2)
    ax.set(xlim=(-.35,len(v)-.65),ylim=(-1.2,1.2))
    ax.set_axis_off()
    return ax


def centre_figure_cell(fig, artists, axes, centre_mm, renderer):
    """Translate one cell's existing artwork; retain data and physical scales.

    Include tick/label extents when centring. Translate overlay artists in
    their data coordinates so PDF/raster export DPI cannot change the offset.
    """
    boxes=[a.get_window_extent(renderer) for a in artists]
    boxes.extend(ax.get_tightbbox(renderer) for ax in axes)
    bounds=Bbox.union([b for b in boxes if b is not None and np.isfinite(b.extents).all()])
    width,height=fig.get_size_inches()*25.4
    target=fig.transFigure.transform([centre_mm[0]/width,1-centre_mm[1]/height])
    shift=target-np.array([(bounds.x0+bounds.x1)/2,(bounds.y0+bounds.y1)/2])
    fraction=shift/np.array([fig.bbox.width,fig.bbox.height])
    for artist in artists:
        if artist.axes is None:
            artist.set_position(np.asarray(artist.get_position())+fraction)
        else:
            inverse=artist.axes.transData.inverted()
            offset=inverse.transform(shift)-inverse.transform([0,0])
            artist.set_transform(Affine2D().translate(*offset)+artist.get_transform())
            artist.set_clip_on(False)
    for ax in axes:
        pos=ax.get_position()
        ax.set_position([pos.x0+fraction[0],pos.y0+fraction[1],pos.width,pos.height])


def fig1():
    """Same physical example, coloured fields and conditional recording densities."""
    hero.CMAP = LinearSegmentedColormap.from_list(
        'soft_blue_coral', ['#4089B4', '#85BED7', '#FAFBF8', '#EFB79B', '#CF735F'])
    hero.CMAP_DRIVE = LinearSegmentedColormap.from_list(
        'signed_drive_blue_coral', ['#589CC2', '#FFFFFF', '#E29673'])
    hero.C_S1, hero.C_S2 = '#169C89', '#AF488F'
    hero.HW_DRIVER = '#435B69'
    hero.STRIP_Y = (14., 18.)
    hero.PHOTO_Y = hero.R_Y[1]+7.
    P, F = hero.pair_model(), hero._fields()
    fig = style.fig_mm(hero.H_MM)
    hero._panel_a(fig, OUT, P)
    source_overlay,drive_key,photo_axis=fig.axes
    hero._panel_b(fig, OUT, F, P)
    # Keep factual condition/axis labels. Panel explanations belong in captions.
    for text in list(fig.texts):
        if text.get_text() in ('Model example', 'Difference field, Δp = p(s₁) − p(s₂)'):
            text.remove()
        elif text.get_text() in ('a', 'b'):
            text.set_text('(' + text.get_text() + ')')
        elif text.get_text() == 'pressure difference, σ':
            text.set_text('Re[p(s₁) − p(s₂)] / σₙ')
    for ax in fig.axes:
        for text in ax.texts:
            if text.get_text() == 'microphones':
                # Keep the pointer tied to the actual projected microphone,
                # independently of the double-headed distance annotation.
                G = hero._model_xy(F)
                mu, mv = hero._Camera()(G['x_mics'], G['y_mics'], hero.MIC_Z)
                text.remove()
                ax.annotate('microphones', xy=(mu[4], mv[4]), xycoords='data',
                    xytext=(0, 12), textcoords='offset points', ha='center',
                    va='bottom', fontsize=7, zorder=25, bbox=hero.LABEL_BOX,
                    arrowprops=dict(arrowstyle='-|>', lw=.6, color=style.INK,
                                    shrinkA=2, shrinkB=2, mutation_scale=6))
            elif text.get_text() == 'two source structures':
                text.set_text('candidate sources')
                text.set_position((hero.A_X+hero.A_W/2, hero.R_Y[0]+hero.ROW_LABEL_DY))
                text.set_horizontalalignment('center')
                text.set_verticalalignment('center')
            elif text.get_text() in ('equal power', 'PHOTO', 'bench array,', 'with the lens', 'without the lens'):
                text.remove()
            elif text.get_text() == 'eight sources':
                text.set_text('sources')
            elif text.get_text() == 'ten cells':
                text.set_text('resonators')
            elif text.get_text() == 'total field, s₁':
                text.set_text(r'Re[$p(s_1)$] / $\sigma_n$')
            elif text.get_text() == 'normalised (−1 … 1)':
                text.set_text('normalized (−1 … 1)')
                text.set_x(hero.A_X+hero.A_W/2)
                text.set_horizontalalignment('center')
            elif text.get_text() == 'Experimental setup':
                text.set_position((hero.A_X+hero.A_W/2,
                                   hero.R_Y[1]+hero.ROW_LABEL_DY))
                text.set_horizontalalignment('center')
                text.set_verticalalignment('center')
            elif text.get_text() in ('drive amplitude,', 'normalized (−1 … 1)',
                                    'equal power',
                                    'bench array,', 'with the lens'):
                text.set_x(hero.A_X+hero.A_W/2)
                text.set_horizontalalignment('center')
    for ax in fig.axes:
        for line in list(ax.lines):
            if len(line.get_xdata()) == 1 and line.get_markersize() in (2.8, 2.6):
                line.remove()
    letter(fig, 126, 1.1, 'c')
    style.tx_mm(fig, 153, 5.2, r'Recording projection $z$', ha='center', fontsize=7)
    Q = loaders.fig01_pair_posterior()
    rows, densities = [], []
    amax = max(np.max(abs(P['s1'])), np.max(abs(P['s2'])))
    density_cells=[]
    for arm, top in zip(('nolens', 'lens'), hero.R_Y):
        r = hero._posterior_row(Q[arm])
        d = P[arm]['d']
        means = (-d/2, d/2)
        assert np.isclose(r['log_odds'], d*d/2)
        # One condition badge governs the field and recording-density row.
        colour = style.C_NOL if arm == 'nolens' else style.C_LENS
        style.tx_mm(fig,105,top+hero.ROW_LABEL_DY,hero.ROW_WORDS[arm],
                    ha='center',va='center',weight='bold',fontsize=7.5,
                    bbox=dict(boxstyle='round,pad=.3',fc='white',ec=colour,lw=.9))
        density_label=style.tx_mm(fig,126,top+18,'Density',rotation=90,
                    ha='center',va='center',fontsize=7)
        cell_axes=[]
        # Both rows use exactly 2 mm per noise standard deviation. The large
        # with-lens separation needs a visibly broken axis, not compressed lobes.
        segments = [(128, 50, -12.5, 12.5)] if arm=='nolens' else [
            (128, 16, means[0]-4, means[0]+4),
            (162, 16, means[1]-4, means[1]+4)]
        for k, (left, width, lo, hi) in enumerate(segments):
            ax = style.ax_mm(fig, left, top+10, width, 17)
            cell_axes.append(ax)
            z = np.linspace(lo, hi, 801)
            p1, p2 = [np.exp(-.5*(z-mu)**2)/np.sqrt(2*np.pi) for mu in means]
            ax.fill_between(z, 0, p1, color=hero.C_S1, alpha=.22, lw=0)
            ax.plot(z, p1, color=hero.C_S1, lw=1.4)
            ax.plot(z, p2, color=hero.C_S2, lw=1.2, ls=hero.LS_S2)
            ax.set(xlim=(lo,hi), ylim=(0,.47), yticks=[], xticks=[0] if arm=='nolens' else [means[k]])
            ax.xaxis.set_major_formatter(lambda v,pos:f'{v:.0f}')
            ax.spines[['left','right','top']].set_visible(False)
            ax.tick_params(labelsize=7,pad=1)
            if lo < means[0] < hi:
                ax.plot([means[0]]*2, [0,.40], ':', color=style.INK, lw=.6)
                ax.plot(means[0],0,'o',ms=3,color=style.INK,clip_on=False)
            if arm=='lens':
                edge = 1 if k==0 else 0
                ax.plot([edge-.035,edge+.035],[-.025,.025],transform=ax.transAxes,
                        color=style.INK,lw=.7,clip_on=False)
                ax.plot([edge+.015,edge+.085],[-.025,.025],transform=ax.transAxes,
                        color=style.INK,lw=.7,clip_on=False)
            for v,a,b in zip(z,p1,p2):
                densities.append(dict(arm=arm,segment=k,z=v,p_z_given_s1=a,p_z_given_s2=b))
        # A source picture and its probability stay together. Neither curve
        # height nor picture opacity encodes the posterior candidate weight.
        glyph_ax=style.ax_mm(fig,126,top+38,53,10)
        cell_axes.append(glyph_ax)
        glyph_ax.set(xlim=(126,179),ylim=(10,0));glyph_ax.set_axis_off()
        for x, name, vals, text in [(129,'s₁',P['s1'],r['t1']), (156,'s₂',P['s2'],r['t2'])]:
            hero._glyph(glyph_ax,x,0,vals,amax,name,1.,arm+'_'+name)
            glyph_ax.text(x+11,4,text.split(' ',1)[1],ha='center',va='top',fontsize=7)
        rows.append((arm, r['w1'], r['w2'], r['log_odds']))
        density_cells.append(([density_label],cell_axes))
    style.tx_mm(fig,153,110,'Probabilities for the mean s₁ record',ha='center',fontsize=7)
    source_data.write(OUT, 'fig01_c_candidates', arm=[r[0] for r in rows],
                      probability_s1=[r[1] for r in rows], probability_s2=[r[2] for r in rows],
                      log_odds_s1=[r[3] for r in rows],
                      condition=['mean recording of s1; lens probabilities drawn as displayed bounds'])
    source_data.write(OUT,'fig01_c_densities',**{k:[r[k] for r in densities] for k in densities[0]})
    # Three column centres and the same two row centres govern every cell.
    # Keep the physical field renderings/camera and density-axis scales fixed.
    for text in fig.texts:
        if text.get_text() in ('(a)','(b)','(c)'):
            centre={'(a)':16.,'(b)':78.,'(c)':152.5}[text.get_text()]
            text.set_x(centre/180);text.set_horizontalalignment('center')
        elif text.get_text()==r'Recording projection $z$':
            text.set_x(152.5/180)
    photo_title=next(t for t in source_overlay.texts if t.get_text()=='Experimental setup')
    upper=[t for t in source_overlay.texts if t is not photo_title]
    upper.extend(source_overlay.collections);upper.extend(source_overlay.patches)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    row_centres=[top+hero.R_H/2 for top in hero.R_Y]
    centre_figure_cell(fig,upper,[drive_key],(16.,row_centres[0]),renderer)
    centre_figure_cell(fig,[photo_title],[photo_axis],(16.,row_centres[1]),renderer)
    for (artists,axes),centre in zip(density_cells,row_centres):
        centre_figure_cell(fig,artists,axes,(152.5,centre),renderer)
    save(fig, 1)


def model_amplitude_data():
    """Original DFT sources at a common per-arm relative budget, both operators."""
    z=thresholds._z()
    operators={tag:operator(tag=='withlens') for tag in COLOURS}
    np.testing.assert_allclose(np.linalg.svd(operators['nolens'],compute_uv=False),
                               z['nolens_sigma']/z['nolens_sigma'][0],rtol=2e-10,atol=1e-14)
    sigma=10**(-float(z['reference_db'])/20)
    arms={}
    for tag,T in operators.items():
        n=T.shape[0];arms[tag]={}
        for mode in thresholds.ROWS:
            phi=thresholds._phi(mode)
            # Same standardized draws across operators; each physical sigma_n
            # follows its own sigma_1 at 64 dB, as in the professor's convention.
            noise=np.random.default_rng(20261104).standard_normal((200,2*n))
            noise=sigma*(noise[:,:n]+1j*noise[:,n:])/np.sqrt(2)
            recordings=T@phi+noise
            fit=fit_known_pattern(T,recordings,phi,sigma)
            arms[tag][mode]=dict(fit,truth=phi,recordings=recordings,
                                errors=abs(fit['amplitude']-1),rms=fit['rms_error'])
    return dict(operator=operators['nolens'],operators=operators,sigma=sigma,
                budget_db=float(z['reference_db']),comparison=arms),arms['nolens']


def model_parameter_choice_data():
    """Physical scales and local controls for the unchanged model benchmark.

    Loss values are sensitivity cases, not measured confidence limits. Use
    the package operator and keep noise-level and main floor cuts distinct.
    """
    from direct import Config, Lens, stacked_matrix
    cfg = Config()
    state, _ = model_amplitude_data()
    T = state['operators']['nolens']
    s = np.linalg.svd(T, compute_uv=False)
    edge = 3*(np.sqrt(T.shape[0])+np.sqrt(T.shape[1]))
    gains = np.array([np.linalg.norm(state['operators'][arm]@row['truth'])
                      for arm,rows in state['comparison'].items() for row in rows.values()])
    # Joint interval: total-noise detection, 2% scalar RMS and exactly rank 3.
    lower = max(20*np.log10(edge/s[2]),
                np.max(20*np.log10(1/(.02*gains))),
                np.max(20*np.log10(np.sqrt(T.shape[0])/gains)))
    upper = 20*np.log10(edge/s[3])
    loss_rows = []
    for loss in (0., cfg.rw_loss, .1):
        c = Config()
        c.rw_loss = loss
        A = stacked_matrix(c, c.freqs(c.f_max_count), Lens(c),
                           with_lens=True, src_resonant=True, src_scatters=True)
        sv = np.linalg.svd(A, compute_uv=False)
        margin = sv/sv[0]*10**(state['budget_db']/20)
        loss_rows.append(dict(loss=loss, noise_rank=int(np.sum(margin>1)),
                              floor_rank=int(np.sum(margin>edge)),
                              weakest_margin=float(margin[-1])))
    return dict(loaded_q=cfg.q_factor/(1+cfg.rw_loss),
                linewidth_pct=100*(1+cfg.rw_loss)/cfg.q_factor,
                loss_share_pct=100*cfg.rw_loss/(1+cfg.rw_loss),
                tone_step_pct=100*(cfg.f_band_hi-cfg.f_band_lo)/(cfg.f_max_count-1),
                microphone_half_angle_deg=float(np.degrees(np.arctan(cfg.screen_half/cfg.mic_standoff))),
                driver_reactive_pct=100*max(abs(cfg.q_src*(cfg.f_res_src_frac**2-f**2))
                    for f in (cfg.f_band_lo,cfg.f_band_hi))/(1+cfg.rw_src),
                budget_interval_db=(np.floor(lower*10+1)/10,
                                    np.ceil(upper*10)/10-.1),
                loss_sweep=loss_rows)


def model_profile_data():
    """Full-profile RMS risk and frozen display draws, with identical scoring in both arms."""
    state,_=model_amplitude_data();z=thresholds._z();order=list(map(str,z['modes']))
    rows={}
    for arm,key in (('nolens','nolens'),('withlens','lens')):
        T=state['operators'][arm];patterns=np.array([thresholds._phi(m) for m in thresholds.ROWS])
        risk=tsvd_profile_risk(T,patterns,state['sigma'],
                              cutoff_factor=3*(np.sqrt(T.shape[0])+np.sqrt(T.shape[1])))
        rows[arm]={m:dict(truth=patterns[i],estimates=z[key+'_return_draws'][order.index(m)],
                         rms=float(risk['rms'][0,i]),rank=int(risk['ranks'][0]),
                         variance=float(risk['variance'][0]),bias_squared=float(risk['bias_squared'][0,i]))
                   for i,m in enumerate(thresholds.ROWS)}
    return rows


def _model_profile_panel(ax, mode, profiles, first, last):
    """Preserve the frozen profiles; replace the component gauge by full-profile RMS."""
    phi=profiles['nolens'][mode]['truth'];peak=max(abs(phi));x=np.arange(1,9)
    ax.axhline(0,xmax=.79,color=style.RULE,lw=.6,zorder=1)
    ax.plot(x,phi/peak,'--o',color=style.C_TRUTH,ms=2,lw=style.LW_DATA,label='Drawn')
    for arm,gx in (('nolens',9.1),('withlens',9.9)):
        row=profiles[arm][mode];est=row['estimates'][0];colour=COLOURS[arm]
        name='No lens' if arm=='nolens' else 'Lens'
        marker='o' if arm=='nolens' else 'D'
        ax.vlines(x,0,est.imag/peak,color=colour,lw=.55,alpha=.55)
        score=f'{row["rms"]:.2f}' if row['rms']>=.1 else f'{row["rms"]:.3f}'
        ax.plot(x,est.real/peak,':' if arm=='nolens' else '-',marker=marker,ms=2.5,
                color=colour,mfc='white' if arm=='nolens' else colour,lw=.9,
                label=f'{name}: RMS {score} {thresholds._glyph(row["rms"]<=PROFILE_RTOL)}')
        ax.plot(gx,row['rms'],marker=marker,ms=3.8,color=colour,
                mfc='white' if arm=='nolens' else colour,zorder=4)
    ax.axvline(8.7,color=style.RULE,lw=.6)
    ax.fill_between((8.7,10.4),PROFILE_RTOL,thresholds.AMP_LIM[1],color=style.C_BAND,lw=0)
    ax.plot((8.7,10.4),(PROFILE_RTOL,PROFILE_RTOL),'--',color=style.INK,lw=.8)
    ax.annotate('0.707',xy=(9.55,PROFILE_RTOL),xytext=(9.55,1.15),
                ha='center',va='bottom',fontsize=7,
                arrowprops=dict(arrowstyle='-',lw=.6,color=style.INK))
    ax.text(9.55,1.72,r'$E_{\rm prof}$',ha='center',va='top',fontsize=7)
    ax.set(xlim=(.5,10.4),ylim=thresholds.AMP_LIM,yticks=[-1,0,1],
           xticks=[*x,9.55],ylabel='Amplitude (real part)')
    if last:
        ax.set_xticklabels([*map(str,x),r'$E_{\rm prof}$']);ax.set_xlabel('Speaker',fontsize=7,labelpad=2)
    else:ax.tick_params(labelbottom=False)
    ax.legend(loc='upper left',fontsize=7,labelspacing=.2,handlelength=1.2,
              borderaxespad=.3,borderpad=.1)
    box(ax)


def fig2():
    """Unchanged detection/scalar panels; common full-profile RMS test in panel c."""
    state,examples=model_amplitude_data()
    profiles=model_profile_data()
    h=thresholds.TOP_Y+3*thresholds.ROW_H+2*thresholds.ROW_GAP+thresholds.BOTTOM
    fig=style.fig_mm(h)
    for x,value in zip((thresholds.A_X,thresholds.B_X,thresholds.C_X),'abc'):
        letter(fig,x,thresholds.HEAD_Y,value)
    style.tx_mm(fig,thresholds.B_X+4,thresholds.SUBHEAD_Y,'B = 64 dB per arm',fontsize=7)
    records=[]
    for i,mode in enumerate(thresholds.ROWS):
        y=thresholds.TOP_Y+i*(thresholds.ROW_H+thresholds.ROW_GAP)
        first,last=i==0,i==2
        thresholds._detection(style.ax_mm(fig,thresholds.A_X,y,thresholds.W,thresholds.ROW_H),mode,first,last)
        _model_profile_panel(style.ax_mm(fig,thresholds.C_X,y,thresholds.W,thresholds.ROW_H),mode,profiles,first,last)
        style.tx_mm(fig,thresholds.ROW_LABEL_X,y+thresholds.ROW_H/2,thresholds.rung_label(mode),
                    rotation=90,va='center',ha='left',fontsize=7)
        ax=style.ax_mm(fig,thresholds.B_X,y,thresholds.W,thresholds.ROW_H)
        ax.set(xlim=(0,1),ylim=(0,1),yticks=[.65,.35],
               yticklabels=['No lens','Lens'],xticks=[0,.5,1])
        ax.text(.04,.96,'True amplitude = 1',transform=ax.transAxes,va='top',fontsize=7)
        ax.text(.04,.06,f'{AMPLITUDE_RTOL:.0%} target: both pass',transform=ax.transAxes,fontsize=7)
        for tag,y0 in [('nolens',.65),('withlens',.35)]:
            s=state['comparison'][tag][mode];rms=100*s['rms'];colour=COLOURS[tag]
            ax.plot([0,rms],[y0,y0],color=colour,lw=1.2)
            ax.plot(rms,y0,'D',ms=4,color=colour)
            ax.text(rms+.035,y0,f'{rms:.2f}%',va='center',fontsize=7)
            for j,A in enumerate(s['amplitude']):
                records.append(dict(arm=tag,mode=mode,draw=j,budget_db=state['budget_db'],
                    amplitude_re=A.real,amplitude_im=A.imag,relative_amplitude_error=abs(A-1),
                    scalar_complex_rms=s['rms'],target=AMPLITUDE_RTOL,amplitude_pass=s['rms']<=AMPLITUDE_RTOL))
        if last:ax.set_xlabel('RMS amplitude error (%)',fontsize=7,labelpad=2)
        else:ax.tick_params(labelbottom=False)
        box(ax)
    thresholds._write_source(OUT,thresholds.ROWS)
    profile_rows=[]
    for arm,rows in profiles.items():
        for mode,row in rows.items():
            estimate=row['estimates'][0];truth=row['truth'];peak=max(abs(truth))
            for speaker in range(8):
                profile_rows.append(dict(arm=arm,mode=mode,speaker=speaker+1,record=0,
                    truth=truth[speaker],estimate_re=estimate[speaker].real,estimate_im=estimate[speaker].imag,
                    displayed_real=estimate[speaker].real/peak,displayed_imag=estimate[speaker].imag/peak,
                    floor_rank=row['rank'],budget_db=state['budget_db'],profile_rms=row['rms'],
                    bias_squared=row['bias_squared'],noise_variance=row['variance'],
                    profile_tolerance=PROFILE_RTOL,profile_pass=row['rms']<=PROFILE_RTOL))
    source_data.write(OUT,'fig02_c_recovery',**{k:[r[k] for r in profile_rows] for k in profile_rows[0]})
    source_data.write(OUT,'fig02_b_estimation',**{k:[r[k] for r in records] for k in records[0]})
    save(fig,2)


def measured_amplitude_budget_data(amplitude_rtol=AMPLITUDE_RTOL):
    """Known-pattern scalar RMS budgets with the existing baseline normalization.

    The budget noise scale is the complex SD per normalized sample, not a
    complete-recording error bound. Detection/recovery keep their own tests.
    The package supplies the scalar gains and exact amplitude thresholds;
    use the original 0.1-dB sampling only for the displayed budget marks.
    """
    start, stop, step = loaders.json_('fig03_measured_budget.json')['budget_sweep_db']
    grid = start + step*np.arange(round((stop-start)/step)+1)
    patterns = budget._drawn()
    result = {}
    for arm, tag in [('nolens', 'nolens'), ('lens', 'withlens')]:
        d = local.measured_blocks(tag)
        H = d['Y'][d['kinds']=='cal'].transpose(1,2,0)
        sigma = np.std(d['Y_baseline_all'],axis=0,ddof=1)
        T = (H/sigma[:,:,None]).reshape(-1,H.shape[-1])
        sweep = pattern_budget_sweep(T,grid,patterns=patterns,amplitude_rtol=amplitude_rtol)
        exact = sweep['thresholds']['amplitude']
        available = 20*np.log10(sweep['singular_values'][0])
        rows = {}
        for i, mode in enumerate(loaders.mode_order()):
            passing = np.flatnonzero(sweep['passed']['amplitude'][:,i])
            if not passing.size:
                raise ValueError(f'Amplitude budget outside sampled range: {arm}/{mode}')
            phi = sweep['patterns'][i]
            fit = fit_known_pattern(T,T@phi,phi,noise_std=1.)
            rows[mode] = dict(pattern=phi,operator=T,gain=fit['gain'],
                rms_at_available=fit['rms_error'],amplitude_rtol=amplitude_rtol,
                budget_exact_db=float(exact[i]),budget_db=float(grid[passing[0]]),
                available_db=float(available))
        result[arm] = rows
    return result


def measured_profile_budget_data(profile_rtol=PROFILE_RTOL, floor_multiplier=3.):
    """Measured operators, assumed white noise, and the shared full-profile RMS test."""
    grid=np.asarray(loaders.measured_budget()['budgets'])
    patterns=budget._drawn();patterns=patterns/np.linalg.norm(patterns,axis=1)[:,None]
    result={}
    for arm, tag in [('nolens', 'nolens'), ('lens', 'withlens')]:
        d = local.measured_blocks(tag)
        H = d['Y'][d['kinds'] == 'cal'].transpose(1, 2, 0)
        sigma = np.std(d['Y_baseline_all'], axis=0, ddof=1)
        T = (H/sigma[:, :, None]).reshape(-1, H.shape[-1])
        factor=floor_multiplier*(np.sqrt(T.shape[0])+np.sqrt(T.shape[1]))
        sweep=profile_recovery_budget(T,grid,patterns,profile_rtol=profile_rtol,cutoff_factor=factor)
        available=tsvd_profile_risk(T,patterns,1.,cutoff_factor=factor)
        result[arm]=dict(sweep,operator=T,available_db=20*np.log10(sweep['singular_values'][0]),
                         rms_at_available=available['rms'][0],rank_at_available=int(available['ranks'][0]))
    return result


def criterion_sensitivity_data():
    """One full-profile metric, varied tolerances and the retained auxiliary capture control."""
    from direct.recover import source_modes
    tolerances={t:measured_profile_budget_data(t) for t in (.5,PROFILE_RTOL,.9)}
    floors={c:measured_profile_budget_data(floor_multiplier=c) for c in (2.,3.,4.)}
    amplitude=measured_amplitude_budget_data()
    result=dict(amplitude_shift_db={tau:20*np.log10(AMPLITUDE_RTOL/tau) for tau in (.02,.04,.08)},arms={})
    for arm,tag in [('nolens','nolens'),('lens','withlens')]:
        raw=local.measured_operator(tag);_,s,vh=np.linalg.svd(raw,full_matrices=False)
        capture=np.sum(abs(vh[s>.03*s[0]]@source_modes(8).T)**2,axis=0)
        result['arms'][arm]=dict(profile_budget_db={t:v[arm]['budget_db'] for t,v in tolerances.items()},
            floor_factor_budget_db={c:v[arm]['budget_db'] for c,v in floors.items()},
            capture_finest={cut:int(np.flatnonzero(capture>=cut)[-1]) for cut in (.3,.5,.7)},
            capture=capture,amplitude_rms=np.array([r['rms_at_available'] for r in amplitude[arm].values()]))
    return result


def fig3():
    """Minimum budgets for the scalar and full-profile RMS criteria."""
    estimates = measured_amplitude_budget_data()
    profiles = measured_profile_budget_data()
    original_heading, original_labels = budget._heading, budget._row_labels
    original_key = budget._budget_key
    original_criteria = budget._criteria
    def headings(fig, x, y, value, text):
        letter(fig, x, y, value)
    def criteria(z):
        values = original_criteria(z)
        for arm in estimates:
            values[arm]['est'] = np.array([estimates[arm][m]['budget_db'] for m in loaders.mode_order()])
            values[arm]['rec'] = profiles[arm]['budget_db']
            values[arm]['first'] = profiles[arm]['first_budget_db']
        return values
    def labels(fig, y_top, names, drawn):
        glyph_x=budget.X0-budget.THUMB_W-budget.THUMB_GAP
        for i, name in enumerate(names):
            centre=y_top+(i+.5)*budget.ROW_MM
            style.tx_mm(fig, glyph_x-budget.LAB_GAP, centre,
                        name, ha='right', va='center', fontsize=7)
            vector_thumbnail(fig,glyph_x,centre-budget.THUMB_H/2,
                             budget.THUMB_W,budget.THUMB_H,drawn[i])
    budget._heading, budget._row_labels = headings, labels
    budget._budget_key = lambda *args: None
    budget._criteria = criteria
    try:
        fig = budget._main(OUT)
    finally:
        budget._heading, budget._row_labels, budget._budget_key = original_heading, original_labels, original_key
        budget._criteria = original_criteria
    for text in list(fig.texts):
        if text.get_text() in (budget.GROUP_A, budget.GROUP_BCD, 'MEASURED'):
            text.remove()
    axes = {a.get_label():a for a in fig.axes}
    axes['fig03_d_channels'].remove()
    axes['fig03_a'].set_xlabel('Received power (dB re floor)',fontsize=7,labelpad=.8)
    for key, task in [('fig03_b', 'detection'),('fig03_c', 'estimation'),('fig03_d', 'recovery')]:
        ax = axes[key]
        ax.set_xlim(budget._budget_lim(original_criteria(loaders.measured_budget()),('det','est','rec')))
        ax.set_xlabel(f'Required {task}\nbudget (dB)',fontsize=7,labelpad=1.5)
        if key=='fig03_d':
            ax.set_xlabel(r'Required recovery'+'\n'+r'budget (dB), $E_{\rm prof}\leq1/\sqrt{2}$',fontsize=7,labelpad=1.5)
        # Labels flank their own lines; the paired marks use the key in a.
        for arm, name, align, dx, colour in (
                ('nolens', 'No lens', 'right', -2, style.C_NOL),
                ('lens', 'With lens', 'left', 2, style.C_LENS)):
            at = next(iter(estimates[arm].values()))['available_db']
            ax.annotate(f'{name}\n{at:.1f} dB', xy=(at, 1.02),
                        xycoords=('data', 'axes fraction'), xytext=(dx, 0),
                        textcoords='offset points', ha=align, va='bottom',
                        fontsize=7, color=colour, annotation_clip=False)
    vectors=budget._drawn();modes=budget.rec._data()['order']
    source_data.write(OUT,'fig03_axis_patterns',mode=np.repeat(modes,8),
                      speaker=np.tile(np.arange(1,9),len(modes)),
                      drive=vectors.ravel(),
                      display_drive=(vectors/np.max(abs(vectors),axis=1)[:,None]).ravel())
    # The supplied renderer also exports retired variance/ridge controls.
    # Current Source Data describe only the displayed three-task comparison.
    z = loaders.measured_budget()
    rows = []
    for arm,label in [('nolens','no lens'),('lens','with lens')]:
        for i,mode in enumerate(modes):
            s = estimates[arm][mode]
            rows.append(dict(arm=label,mode=mode,label=budget.rec.pattern_label(mode),
                available_db=s['available_db'],floor_rank=int(z[arm+'_rank_ref']),
                B_detected_db=float(z[arm+'_B_detected'][i]),B_estimated_db=s['budget_db'],
                B_estimated_exact_db=s['budget_exact_db'],amplitude_gain=s['gain'],
                amplitude_rms_at_available=s['rms_at_available'],amplitude_rtol=s['amplitude_rtol'],
                margin_unit_norm_db=float(z[arm+'_margin_ref'][i]),
                B_tsvd_profile_rms_db=float(profiles[arm]['budget_db'][i]),
                B_tsvd_profile_rms_first_db=float(profiles[arm]['first_budget_db'][i]),
                profile_rms_at_available=float(profiles[arm]['rms_at_available'][i]),
                profile_tolerance=PROFILE_RTOL,
                profile_pass_at_available=profiles[arm]['rms_at_available'][i]<=PROFILE_RTOL))
    source_data.write(OUT,'fig03_bcd_budgets',**{k:[r[k] for r in rows] for k in rows[0]})
    save(fig, 3)


def measured_amplitude_data():
    """Five true records per pattern; all-channel, prior-free scalar fitting."""
    archive=np.load(DATA/'scalar_ladder_phasors.npz',allow_pickle=False)
    R=loaders.npz('fig04_rows.npz')
    out={}
    for arm,tag in [('A','nolens'),('B','withlens')]:
        d=local.measured_blocks(tag)
        cal=np.flatnonzero(d['kinds']=='cal')
        H=d['Y'][cal].transpose(1,2,0).reshape(-1,8)
        sigma1=np.linalg.svd(H,compute_uv=False)[0]
        out[arm]={}
        for j,mode in enumerate(PATTERNS):
            phi=np.asarray(R['truth_profiles'][j])
            if mode=='cos_p4':
                Y=d['Y'][d['kinds']=='image:staggered']
            else:
                Y=archive[arm+'_'+mode]-d['Y_baseline_all'].mean(axis=0)
            fitted=fit_known_pattern(H,Y.reshape(5,-1),phi)
            # Same conditional recording-error assumption as the earlier local
            # amplitude figure. This is neither a noise SD nor a measured bound.
            epsilon=local.MEAS_REL_FLOOR*sigma1*np.linalg.norm(phi)
            out[arm][mode]=dict(fitted,truth=phi,recordings=Y,
                errors=np.abs(fitted['amplitude']-1),assumed_error_level=epsilon,
                conditional_limit=epsilon/fitted['gain'])
    return out


def measured_profile_data():
    """Five-record RMS full-profile scores using the unchanged first weights/rank five."""
    fits=measured_amplitude_data();frozen=loaders.fig04_tsvd_complex();result={}
    for arm,tag in [('A','nolens'),('B','withlens')]:
        block=local.measured_blocks(tag)
        H=block['Y'][block['kinds']=='cal'].transpose(1,2,0).reshape(-1,8)
        weights=1/abs(block['Y_baseline_all'][0]).reshape(-1)
        s=np.linalg.svd(weights[:,None]*H,compute_uv=False)
        inverse,rank=truncated_pinv(weights[:,None]*H,(s[4]+s[5])/2)
        if rank!=5:raise ValueError('The recorded comparison requires hard rank five')
        result[arm]={}
        for mode,row in fits[arm].items():
            estimates=(inverse@(weights[:,None]*row['recordings'].reshape(5,-1).T)).T
            np.testing.assert_allclose(estimates,frozen[f'ret45_{arm}_{mode}_rank5'][0],atol=1e-10)
            result[arm][mode]=dict(truth=row['truth'],estimates=estimates,
                                  rms=relative_profile_rms(estimates,row['truth']))
    return result


def recorded_setting_sensitivity_data():
    """Rebuild the existing rank/weight controls from separate recordings.

    Nine weights reuse the same five records; they are analysis settings, not
    independent experiments. RMS scores include the full complex error.
    """
    from direct.recover import truncated_pinv
    frozen = loaders.fig04_tsvd_complex()
    fits = measured_amplitude_data()
    scores = {}
    max_difference = 0.
    for arm,tag in (('A','nolens'),('B','withlens')):
        d = local.measured_blocks(tag)
        H = d['Y'][d['kinds']=='cal'].transpose(1,2,0).reshape(-1,8)
        for rank in (4,5,6):
            for mode in PATTERNS:
                errors = []
                for j,b in enumerate(d['Y_baseline_all']):
                    w = 1/abs(b).reshape(-1)
                    sv = np.linalg.svd(w[:,None]*H,compute_uv=False)
                    K, retained = truncated_pinv(w[:,None]*H,(sv[rank-1]+sv[rank])/2)
                    if retained != rank:
                        raise ValueError('The hard-rank control did not retain its stated rank')
                    estimates = (K@(w[:,None]*fits[arm][mode]['recordings'].reshape(5,-1).T)).T
                    max_difference = max(max_difference,float(np.max(abs(
                        estimates-frozen[f'ret45_{arm}_{mode}_rank{rank}'][j]))))
                    a = fits[arm][mode]['truth']
                    errors.append(np.linalg.norm(estimates-a,axis=1)/np.linalg.norm(a))
                scores[arm,rank,mode] = np.sqrt(np.mean(np.asarray(errors)**2,axis=1))
    return dict(max_frozen_difference=max_difference,
                improvement_counts={rank:[int(np.sum(scores['A',rank,m]>scores['B',rank,m]))
                                          for m in PATTERNS] for rank in (4,5,6)})


def amplitude_tolerance_data():
    """Retrospective tolerance from all retained five-repeat scalar-fit cases.

    The 54 EXP06 cases and four separately archived Fourier-sine cases use
    unweighted complex fits, true A=1 and literal acquired drives. Case RMS
    includes calibration mismatch. The nearest-rank 95th percentile is a
    descriptive coverage convention, not a population confidence statement.
    """
    rows=[]
    for tag in ('nolens','withlens','nolens_b6'):
        d=local.measured_blocks(tag);H=local.measured_H_per_tone(tag)
        for kind in sorted(k for k in set(d['kinds']) if str(k).startswith('image:')):
            ids=np.flatnonzero(d['kinds']==kind)
            drives=d['D'][ids]
            if len(ids)!=5 or not np.all(drives==drives[0]):
                raise ValueError('Tolerance calibration requires five repeats of one literal drive')
            h=np.einsum('kmn,kn->km',H,drives[0]).reshape(-1)
            fit=fit_known_pattern(h[:,None],d['Y'][ids].reshape(5,-1),np.ones(1))
            errors=np.abs(fit['amplitude']-1)
            rows.append(dict(arm=tag,pattern=str(kind),errors=errors,
                             rms=float(np.sqrt(np.mean(errors**2)))))
    for arm,modes in measured_amplitude_data().items():
        for mode in ('sin_p1','sin_p2'):
            errors=modes[mode]['errors']
            rows.append(dict(arm='nolens' if arm=='A' else 'withlens',pattern=mode,
                             errors=errors,rms=float(np.sqrt(np.mean(errors**2)))))
    rms=np.array([row['rms'] for row in rows]);coverage=.95;step=.01
    rank=int(np.ceil(coverage*len(rms)))
    cutoff=float(np.sort(rms)[rank-1])
    selected=float(step*np.ceil(cutoff/step))
    return dict(cases=rows,case_rms=rms,coverage=coverage,rank=rank,cutoff=cutoff,
                step=step,selected=selected,count=len(rows),
                recording_count=sum(len(row['errors']) for row in rows),
                passing_count=int(np.count_nonzero(rms<=selected)))


def fig4():
    """Unchanged single profiles/powers; shared RMS profile criterion from five records."""
    R=loaders.npz('fig04_rows.npz')
    fits=measured_amplitude_data();profiles=measured_profile_data();fig=style.fig_mm(140)
    letter(fig, 3, 3, 'a');letter(fig,3,47,'b');letter(fig,3,91,'c')
    records=[];power_rows=[];profile_rows=[]
    for col,(arm,tag) in enumerate([('A','nolens'),('B','withlens')]):
        colour=COLOURS[tag];left=16+89*col
        style.tx_mm(fig,left+37,1,'without lens' if arm=='A' else 'with lens',ha='center')
        a=style.ax_mm(fig,left,9,71,29)
        a.fill_between(np.arange(1,7),*R[arm+'_det_floor_band_db'],color=style.C_BAND,lw=0)
        a.axhline(0,color=style.INK3,ls='--',lw=.8)
        for j,mode in enumerate(PATTERNS):
            a.plot(np.arange(1,7),R[arm+'_det_rx_raw_db_median'][j],
                   marker=['o','s','v'][j],ms=3,color=colour,ls=['-',':','--'][j],label=LABELS[j])
            for mic in range(6):
                power_rows.append(dict(arm=arm,mode=mode,microphone=mic+1,
                    power_db=R[arm+'_det_rx_raw_db_median'][j,mic],
                    floor_lo_db=R[arm+'_det_floor_band_db'][0,mic],floor_hi_db=R[arm+'_det_floor_band_db'][1,mic]))
        a.set(ylim=(-6,45),xticks=range(1,7),xlabel='Microphone',ylabel='Power (dB re floor)' if col==0 else '')
        a.legend(loc='upper left',fontsize=7,handlelength=1.3,labelspacing=.2,borderpad=.1)
        box(a)
        # Different labelled axes expand the observed-error range without hiding
        # the larger conditional limits or implying one continuous horizontal scale.
        bleft=left+11
        b=style.ax_mm(fig,bleft,53,41,28)
        bounds=style.ax_mm(fig,bleft+46,53,14,28)
        b.axvspan(100*AMPLITUDE_RTOL,10,color='#F5E6DF',zorder=0)
        b.axvline(100*AMPLITUDE_RTOL,color=style.INK,ls='--',lw=.9)
        b.set(ylim=(2.7,-.6),xlim=(-.15,10),yticks=range(3),
              yticklabels=['']*3,xticks=[0,2,4,6,8,10])
        bounds.set(ylim=(2.7,-.6),xlim=(0,65),yticks=[],xticks=[0,30,60])
        for j,mode in enumerate(PATTERNS):
            s=fits[arm][mode];A=s['amplitude']
            centre=53+(j+.6)/3.3*28
            glyph_x=bleft-9.2
            style.tx_mm(fig,glyph_x-1,centre,LABELS[j],ha='right',va='center',fontsize=7)
            vector_thumbnail(fig,glyph_x,centre-1.7,8,3.4,s['truth'])
            errors=100*s['errors'];median=np.median(errors)
            limit=100*s['conditional_limit']
            bounds.hlines(j,0,limit,color=colour,lw=1.1,alpha=.7)
            bounds.plot(limit,j,'|',color=colour,ms=7,mew=1.2)
            bounds.text(limit+(1 if limit<35 else -1),j-.16,f'{limit:.0f}%',
                        ha='left' if limit<35 else 'right',va='bottom',fontsize=7)
            b.scatter(errors,j+np.linspace(-.12,.12,5),s=15,color=colour,
                      marker='o' if col==0 else 'D',alpha=.65)
            b.text(5,j+.34,f'median {median:.1f}%',ha='left',va='center',fontsize=7)
            for i,value in enumerate(A):
                records.append(dict(arm=arm,mode=mode,record=i,amplitude_re=value.real,amplitude_im=value.imag,
                                    complex_error=abs(value-1),within_4_percent=abs(value-1)<=AMPLITUDE_RTOL,
                                    assumed_error_level=s['assumed_error_level'],
                                    conditional_limit=s['conditional_limit'],
                                    conditional_pass=s['conditional_limit']<=AMPLITUDE_RTOL))
        b.set_xlabel('Amplitude error (%)',fontsize=7)
        bounds.set_xlabel('Limit (%)',fontsize=7)
        b.text(100*AMPLITUDE_RTOL,1.05,f'{AMPLITUDE_RTOL:.0%} target',
               transform=b.get_xaxis_transform(),ha='center',fontsize=7)
        bounds.text(.5,1.05,r'$\epsilon/g$',transform=bounds.transAxes,ha='center',fontsize=7)
        box(b);box(bounds)
        for j,mode in enumerate(PATTERNS):
            c=style.ax_mm(fig,left+24.1*j,98,22.8,29)
            row=profiles[arm][mode];truth=row['truth'];est=row['estimates'][0]
            peak=max(abs(truth));x=np.arange(1,9)
            c.plot(x,truth/peak,'--o',color=style.C_TRUTH,ms=2,lw=.8)
            c.plot(x,est.real/peak,':o' if arm=='A' else '-D',color=colour,ms=2.2,lw=1,mfc='white' if arm=='A' else colour)
            for record,estimate in enumerate(row['estimates']):
                record_error=relative_profile_rms(estimate,truth)
                for speaker in range(8):
                    profile_rows.append(dict(arm=arm,mode=mode,speaker=speaker+1,truth=truth[speaker],
                        estimate_re=estimate[speaker].real,estimate_im=estimate[speaker].imag,
                        rank=5,baseline_weight=0,record=record,displayed=record==0,
                        full_relative_error=record_error,profile_rms=row['rms'],
                        profile_tolerance=PROFILE_RTOL,profile_pass=row['rms']<=PROFILE_RTOL))
            c.set(xlim=(.6,8.4),ylim=(-1.7,1.7),xticks=[1,4,8],yticks=[-1,0,1])
            if j>0:c.set_yticklabels([])
            cl=left+24.1*j
            style.tx_mm(fig,cl+.2,93.5,LABELS[j],va='center',fontsize=7)
            vector_thumbnail(fig,cl+14,91.8,8,3.4,truth)
            c.text(.97,.97,f'RMS {row["rms"]:.2f} {thresholds._glyph(row["rms"]<=PROFILE_RTOL)}',transform=c.transAxes,ha='right',va='top',fontsize=7,
                   bbox=dict(fc='white',ec='none',alpha=.9,pad=.5))
            if j==0 and col==0:c.set_ylabel('Drive / drawn peak',fontsize=7)
            c.set_xlabel('Speaker',fontsize=7);box(c)
    style.tx_mm(fig,90,136,r'RMS relative profile error: $E_{\rm prof}\leq1/\sqrt{2}$',ha='center',fontsize=7)
    source_data.write(OUT,'fig04_scalar_fits',**{k:[r[k] for r in records] for k in records[0]})
    source_data.write(OUT,'fig04_received_powers',**{k:[r[k] for r in power_rows] for k in power_rows[0]})
    source_data.write(OUT,'fig04_source_profiles',**{k:[r[k] for r in profile_rows] for k in profile_rows[0]})
    save(fig,4)


def prior_data():
    """Flat data-only baseline, fixed physical patterns and full covariance updates.

    Model: two unknown coefficients in one lens-reference SVD basis, common
    relative budget 24 dB. One prescribed standardized displacement, not a
    search over noisy examples. Measured: existing source-class example and
    its held-out-class empirical error model; all eight real drives are free.
    """
    M={tag:real_operator(operator(tag=='withlens')) for tag in COLOURS}
    V=np.linalg.svd(M['withlens'],full_matrices=False)[2].T[:,[2,4]]
    for j in range(2):
        if V[np.argmax(abs(V[:,j])),j]<0:V[:,j]*=-1
    truth=np.ones(2);sigma=10**(-24/20);result={'model':{},'measured':{}}
    for tag,A in M.items():
        W=A@V;U,s,vh=np.linalg.svd(W,full_matrices=False)
        S=(vh.T*((sigma*sigma/2)/(s*s)))@vh
        z=truth+np.linalg.cholesky(S)@np.array([.4,-.55])
        est={'none':z};post={}
        for key,rho in [('correct',.9),('wrong',-.9)]:
            C=np.array([[1,rho],[rho,1]])
            r=gaussian_statistic_partial_prior(z,S,[0,1],np.zeros(2),C)
            est[key]=r['estimate'];post[key]=r['posterior']
        result['model'][tag]=dict(basis=V,truth=truth,data_cov=S,estimates=est,posterior=post,
            profiles={'truth':V@truth,**{k:V@v for k,v in est.items()}})
    result['measured']=local.measured_probabilistic_prior_table()
    return result


def prior_neural_cases():
    """Sufficient LS observations for the CURRENT two-coefficient model example.

    This is not the older 64-dB v5/v7 control. Training sources and noise are
    sampled independently; the fixed plotted truth is only an evaluation point.
    """
    result={}
    for arm,row in prior_data()['model'].items():
        result[arm]=dict(basis=row['basis'],truth=row['truth'],
            observation=row['estimates']['none'],noise_covariance=row['data_cov'],
            priors={key:dict(mean=np.zeros(2),covariance=np.array([[1.,rho],[rho,1.]]))
                    for key,rho in (('correct',.9),('wrong',-.9))},
            analytic={key:row['estimates'][key] for key in ('correct','wrong')})
    return result


def prior_neural_fingerprint(cases):
    """Pin both physical bases, likelihoods, priors and plotted observations."""
    digest=hashlib.sha256()
    for arm in COLOURS:
        row=cases[arm]
        for key in ('basis','truth','observation','noise_covariance'):
            digest.update(np.asarray(row[key],dtype='<f8').tobytes())
        for key in ('correct','wrong'):
            for name in ('mean','covariance'):
                digest.update(np.asarray(row['priors'][key][name],dtype='<f8').tobytes())
    return digest.hexdigest()


def prior_neural_results():
    """Actual trained outputs; reject changes to the recorded physical example.

    SVD roundoff varies between Python/BLAS builds, so its float64 byte hash
    is not a portable numerical identity. When that hash differs, require
    agreement with the independent, frozen six-significant-digit Source Data.
    The original checkpoint fingerprints and weights remain unchanged.
    """
    path=ROOT/'notebooks/models/nn_prior_current/results.json'
    result=json.loads(path.read_text(encoding='utf-8'))
    cases=prior_neural_cases()
    if result['geometry_sha256']!=prior_neural_fingerprint(cases):
        reference=json.loads((path.parent/'geometry_reference.json').read_text(encoding='utf-8'))
        if reference['geometry_sha256']!=result['geometry_sha256']:
            raise ValueError('Figure 5 numerical reference belongs to a different training record')
        try:
            for arm,row in cases.items():
                frozen=reference['arms'][arm]
                for key in ('basis','truth','observation','noise_covariance'):
                    np.testing.assert_allclose(row[key],frozen[key],
                        rtol=reference['rtol'],atol=reference['atol'])
                for key,prior in row['priors'].items():
                    for name in ('mean','covariance'):
                        np.testing.assert_array_equal(prior[name],frozen['priors'][key][name])
        except (AssertionError,KeyError) as error:
            raise ValueError('Current Figure 5 neural results are stale; train with --current-figure') from error
    return result


def ellipse(ax,centre,cov,colour,*,fill=False,lw=.8,alpha=1):
    v,w=np.linalg.eigh(cov);order=np.argsort(v)[::-1];v,w=v[order],w[:,order]
    a=Ellipse(centre,4*np.sqrt(max(v[0],0)),4*np.sqrt(max(v[1],0)),
              angle=np.degrees(np.arctan2(w[1,0],w[0,0])),
              fc=colour if fill else 'none',ec=colour,lw=lw,alpha=alpha)
    ax.add_patch(a)


def pattern_glyph(ax,values,orientation='x'):
    """Centre peak-normalized views of unit-norm patterns at their own axes."""
    v=np.asarray(values)/max(abs(values));x=np.arange(8)
    if orientation=='x':
        ax.plot(x,v,'o-',ms=1.8,lw=.7,color=style.C_LENS)
        ax.set(xlim=(-.35,7.35),ylim=(-1.1,1.1))
    else:
        ax.plot(v,x,'o-',ms=1.8,lw=.7,color=style.C_LENS)
        ax.set(xlim=(-1.1,1.1),ylim=(-.35,7.35))
    ax.set_axis_off()


def prior_shift(ax,state,limits,zoom=False,neural=None,prior='correct',label_offset=None):
    """Arrow follows the actual update; inset magnifies both physical axes equally."""
    start=state['estimates']['none'];end=state['estimates'][prior]
    colour=PRIOR_COLOURS[prior]
    low,high=limits;span=high-low
    truth=state['profiles']['truth'];norm=np.linalg.norm(truth)
    before=np.linalg.norm(state['profiles']['none']-truth)/norm
    after=np.linalg.norm(state['profiles'][prior]-truth)/norm
    change_pp=100*(after-before)
    label=f'{change_pp:+.1f} pp'.replace('-','−')
    def arrow(target,size):
        target.annotate('',xy=end,xytext=start,
            arrowprops=dict(arrowstyle='->',color=colour,lw=1,
                            shrinkA=size,shrinkB=size,mutation_scale=9),zorder=5)
    if np.linalg.norm((end-start)/span*np.array([34,40]))>3:
        arrow(ax,1.5)
        if not zoom:
            offset=label_offset if label_offset is not None else ((-6,0) if prior=='correct' else (6,0))
            ax.annotate(label,xy=(start+end)/2,xytext=offset,textcoords='offset points',
                        ha='right' if prior=='correct' else 'left',va='center',fontsize=7,zorder=6)
    result=dict(from_x=start[0],from_y=start[1],to_x=end[0],to_y=end[1],
                relative_profile_error_before=before,relative_profile_error_after=after,
                error_change_percentage_points=change_pp,
                zoom_magnification=1)
    if not zoom:return result
    position=(.43,.70) if prior=='correct' else (.43,.06)
    relative_size=np.array([.55,.27] if prior=='correct' else [.55,.40])
    points=np.array([start,end]+([neural] if neural is not None else []))
    extent=np.maximum(np.ptp(points,axis=0),1e-10)
    # Exact displayed magnification, accounting for the parent/inset dimensions.
    usable,step=(np.array([1/1.8,1/1.8]),5) if prior=='correct' else (np.array([.76,.50]),.5)
    maximum_mag=np.min(span*relative_size*usable/extent)
    mag=max(1,step*int(maximum_mag//step))
    zspan=span*relative_size/mag;centre=(points.min(0)+points.max(0))/2
    # Leave space below the wrong-prior endpoints for the error-change label.
    zlow=centre-zspan*np.array([.5,.5 if prior=='correct' else .55]);zhigh=zlow+zspan
    detail=ax.inset_axes([*position,*relative_size],zorder=8)
    detail.set(xlim=(zlow[0],zhigh[0]),ylim=(zlow[1],zhigh[1]),xticks=[],yticks=[])
    detail.set_facecolor('white')
    for value,key,marker in [(start,'none','s'),(end,prior,'^' if prior=='correct' else 'v')]:
        detail.plot(*value,marker=marker,color=PRIOR_COLOURS[key],ms=2.6,ls='',zorder=6)
    if neural is not None:
        detail.plot(*neural,marker='o',mfc='none',mec=style.C_NN,ms=3.4,ls='',mew=.7,zorder=7)
    arrow(detail,1.1)
    detail.text(.97,.97,f'×{mag:g}',transform=detail.transAxes,ha='right',va='top',fontsize=7)
    detail.text(.5,.05,label,transform=detail.transAxes,ha='center',va='bottom',fontsize=7)
    for spine in detail.spines.values():spine.set(linewidth=.6,color=style.C_FRAME)
    ax.add_patch(Rectangle(zlow,*zspan,fill=False,ec=style.C_FRAME,lw=.6,zorder=4))
    corner=((zhigh if prior=='correct' else zlow)-low)/span
    anchor=position if prior=='correct' else (position[0],position[1]+relative_size[1])
    ax.plot([corner[0],anchor[0]],[corner[1],anchor[1]],transform=ax.transAxes,
            color=style.C_FRAME,lw=.6,zorder=3)
    result.update(zoom_magnification=mag,zoom_xmin=zlow[0],zoom_xmax=zhigh[0],
                  zoom_ymin=zlow[1],zoom_ymax=zhigh[1])
    return result


def fig5():
    data=prior_data();neural=prior_neural_results();fig=style.fig_mm(136)
    # Compact the gutters while retaining each panel's physical plotting scale.
    fig.set_size_inches(168/25.4,136/25.4)
    markers={'none':'s','correct':'^','wrong':'v'}
    names={'none':'Data only','correct':'Compatible prior','wrong':'Wrong prior'}
    rows=[];profile_rows=[];basis_rows=[];cov_rows=[];shift_rows=[]
    neural_rows=[];neural_profiles=[]
    limits={}
    for group,arms in data.items():
        lows=[];highs=[]
        for state in arms.values():
            pts=np.array([state['truth'],*state['estimates'].values()]);sd=2*np.sqrt(np.diag(state['data_cov']))
            lows.append(np.minimum(pts.min(0),state['estimates']['none']-sd))
            highs.append(np.maximum(pts.max(0),state['estimates']['none']+sd))
        lo,hi=np.min(lows,axis=0),np.max(highs,axis=0)
        span=np.maximum(hi-lo,[.25,.35]);centre=(lo+hi)/2
        limits[group]=(centre-.62*span,centre+.62*span)
    for row,group in enumerate(['model','measured']):
        for col,tag in enumerate(COLOURS):
            state=data[group][tag];left=18+82*col;top=10+64*row
            letter(fig,left-10,top-7,chr(97+row*2+col))
            style.tx_mm(fig,left+65,top-7,('24 dB / ' if row==0 else '')+
                        ('without lens' if col==0 else 'with lens'),ha='right',fontsize=7)
            a=style.ax_mm(fig,left,top,34,40)
            base=state['estimates']['none'];truth=state['truth'];S=state['data_cov']
            for i in range(2):
                for j in range(2):
                    cov_rows.append(dict(group=group,arm=tag,i=i,j=j,covariance=S[i,j],
                        data_centre=base[i],truth=truth[i]))
            for speaker in range(8):
                basis_rows.append(dict(group=group,arm=tag,speaker=speaker+1,
                    x_pattern=state['basis'][speaker,0],y_pattern=state['basis'][speaker,1]))
            ellipse(a,base,S,'#BFD6E2',fill=True,alpha=.60)
            lo,hi=limits[group]
            a.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
            for key,value in state['estimates'].items():
                a.plot(*value,marker=markers[key],color=PRIOR_COLOURS[key],ms=3,ls='',zorder=6)
            learned=neural['arms'][tag]['neural'] if group=='model' else {}
            for value in learned.values():
                a.plot(*value,marker='o',mfc='none',mec=style.C_NN,ms=3.8,ls='',mew=.7,zorder=7)
            a.plot(*truth,'*',ms=5,color=style.INK,zorder=7)
            shift_rows.append(dict(group=group,arm=tag,prior='correct',
                **prior_shift(a,state,limits[group],zoom=(tag=='withlens'),neural=learned.get('correct'))))
            shift_rows.append(dict(group=group,arm=tag,prior='wrong',
                **prior_shift(a,state,limits[group],prior='wrong',zoom=(tag=='withlens'),
                    neural=learned.get('wrong'))))
            a.locator_params(axis='both',nbins=3)
            # Glyph centres coincide with their own axis midpoint. Keep the
            # glyph next to the ticks and put its text label outside it.
            a.set_xlabel('Pattern 1',fontsize=7,labelpad=24)
            a.set_ylabel('Pattern 2',fontsize=7,labelpad=21)
            gx=a.inset_axes([.285,-.215,.43,.13]);pattern_glyph(gx,state['basis'][:,0])
            gy=a.inset_axes([-.27,.285,.13,.43]);pattern_glyph(gy,state['basis'][:,1],'y')
            box(a)
            profiles=state['profiles'];scale=max(abs(profiles['truth']))
            for j,key in enumerate(['none','correct','wrong']):
                p=style.ax_mm(fig,left+41,top+j*16,24,12)
                p.plot(range(1,9),profiles['truth']/scale,'--',lw=.8,color=style.C_TRUTH)
                p.plot(range(1,9),profiles[key]/scale,marker=markers[key],ms=2.2,lw=.8,color=PRIOR_COLOURS[key])
                if key in learned:
                    coeff=np.asarray(learned[key]);profile=state['basis']@coeff
                    p.plot(range(1,9),profile/scale,':',marker='o',mfc='none',ms=2.8,
                           lw=1,mew=.7,color=style.C_NN,zorder=5)
                    neural_rows.append(dict(arm=tag,prior=key,seed=neural['figure_seed'],
                        x=coeff[0],y=coeff[1],analytic_x=state['estimates'][key][0],
                        analytic_y=state['estimates'][key][1],
                        relative_profile_error=np.linalg.norm(profile-profiles['truth'])/np.linalg.norm(profiles['truth'])))
                    for speaker in range(8):
                        neural_profiles.append(dict(arm=tag,prior=key,speaker=speaker+1,
                            truth=profiles['truth'][speaker],analytic=profiles[key][speaker],neural=profile[speaker]))
                err=np.linalg.norm(profiles[key]-profiles['truth'])/np.linalg.norm(profiles['truth'])
                p.set(xlim=(.6,8.4),ylim=(-1.9,1.9),xticks=[1,8],yticks=[-1,0,1])
                p.text(.01,1.08,names[key],transform=p.transAxes,fontsize=7)
                p.text(.98,.03,f'err {err:.1%}',ha='right',transform=p.transAxes,fontsize=7)
                if j<2:p.tick_params(labelbottom=False)
                else:p.set_xlabel('Speaker',fontsize=7)
                box(p)
                rows.append(dict(group=group,arm=tag,estimator=key,x=state['estimates'][key][0],
                                 y=state['estimates'][key][1],shown_profile_relative_error=err))
                for speaker in range(8):
                    profile_rows.append(dict(group=group,arm=tag,estimator=key,speaker=speaker+1,
                        truth=profiles['truth'][speaker],estimate=profiles[key][speaker]))
    handles=[Line2D([],[],marker='*',ls='',color=style.INK,ms=5,label='True source')]
    handles += [Line2D([],[],marker=markers[k],ls='',color=PRIOR_COLOURS[k],ms=3,label=names[k]) for k in markers]
    handles += [Line2D([],[],marker='o',mfc='none',ls=':',color=style.C_NN,ms=3.8,label='Neural (model)')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.006),ncol=5,fontsize=7,frameon=False,
               columnspacing=1,handletextpad=.3)
    source_data.write(OUT,'fig05_prior_comparison',**{k:[r[k] for r in rows] for k in rows[0]})
    for name,values in [('fig05_source_profiles',profile_rows),('fig05_axis_patterns',basis_rows),('fig05_data_covariances',cov_rows)]:
        source_data.write(OUT,name,**{k:[r[k] for r in values] for k in values[0]})
    shift_keys=list(dict.fromkeys(k for r in shift_rows for k in r))
    source_data.write(OUT,'fig05_prior_shifts',**{k:[r.get(k,'') for r in shift_rows] for k in shift_keys})
    for name,values in [('fig05_neural_comparison',neural_rows),('fig05_neural_profiles',neural_profiles)]:
        source_data.write(OUT,name,**{k:[r[k] for r in values] for k in values[0]})
    save(fig,5)


def write_metadata():
    """Keep figure notes and numerical provenance beside the adopted assets."""
    text='# Figure revision for author review — 2026-11-06\n\n'
    text+='Build: `py -B paper/scripts/figure_revision.py` from the repository root.\n\n'
    for n,caption in CAPTIONS.items():
        text+=f'## Figure {n}\n\n{caption}\n\n'
    (OUT/'captions.md').write_text(text.rstrip()+'\n',encoding='utf-8')
    inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.iterdir() if p.is_file()}
    manifest=dict(upstream_repository='https://github.com/wavelabnu/metalens-paper',
        upstream_commit='eb00db8387dc7965de1d170247d3316df09b90f6',
        inputs_sha256=inputs,
        source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        figure1_discriminant="z = Re[delta_y^H (y-y_bar)] / ((sigma/sqrt(2))*norm(delta_y)); delta_y=T(s2-s1)",
        figure1_layout="one bordered condition badge per field/density row; numerical separation and mean summaries omitted; object labels and inset field quantity distinguished; microphones pointer vertically downward; decorative row dividers removed; columns centred at 16/78/152.5 mm and cell contents at common row centres 34.5/87.5 mm",
        model_noise_seed=20261104,model_amplitude_draws=200,
        model_amplitude_sources=list(thresholds.ROWS),model_amplitude_budget_db=64,
        figure2_preserved_numerics=["detection", "scalar amplitude", "frozen no-lens profile draw"],
        figure2_amplitude_comparison="both arms at their own sigma1-normalized 64 dB",
        figure4_conditional_level="assumed recording-error bound epsilon = 0.03*sigma1*norm(source)",
        figure4_amplitude_axes="wide recorded errors 0-10 percent; separate aligned conditional limits 0-65 percent; all values preserved",
        figure5_insets="separate compatible- and wrong-prior shifts; same estimates, uniform physical display magnification, no point displacement",
        figure5_layout="168 by 136 mm; column pitch 82 mm; row pitch 64 mm; unchanged 34 by 40 mm coefficient axes",
        figure5_arrow_labels="data-only to compatible and wrong priors; change in full-source relative profile error, percentage points",
        source_glyphs="eight actual drive components; each glyph normalized to its own peak for display",
        model_prior_reference='with-lens real-stack SVD, one-based directions 3 and 5',
        model_prior_budget_db=24,model_prior_standardized_displacement=[.4,-.55],
        model_neural_results_sha256=hashlib.sha256((ROOT/'notebooks/models/nn_prior_current/results.json').read_bytes()).hexdigest(),
        model_neural_geometry_sha256=prior_neural_results()['geometry_sha256'],
        measured_prior_target='image:pair_gap_0_2, all five repeats',
        measured_profile_display_selection='C8 hard rank 5, baseline 0, record 0 in every cell; RMS uses all five records',
        figure3='full-profile RMS recovery budgets recalculated from EXP06; detection/amplitude unchanged',
        reconstruction_metric='RMS relative full complex profile error; no phase or amplitude refitting',
        reconstruction_tolerance=PROFILE_RTOL,
        reconstruction_mean_square_tolerance=PROFILE_RTOL**2,
        reconstruction_averaging='exact noise expectation in Figures 2/3; five acquired records in Figure 4',
        figure3_amplitude_rtol=AMPLITUDE_RTOL,
        amplitude_target_mean_square_error=AMPLITUDE_RTOL**2,
        amplitude_tolerance_selection={k:v for k,v in amplitude_tolerance_data().items()
                                       if k not in ('cases','case_rms')},
        figure3_amplitude_noise='assumed independent proper-complex noise after baseline standard-deviation normalization',
        figure3_amplitude_budget_grid_db=loaders.json_('fig03_measured_budget.json')['budget_sweep_db'],
        figure4_error_level_provenance='0.03 is a selected scale motivated by EXP06 recording-prediction-error medians; not a universal bound')
    (OUT/'source_data/provenance.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


def extract_raw(raw_root):
    """Reproduce the retained sine phasors from the 20 hash-pinned raw files.

    Raw root is a caller-supplied staging folder with A_sin_p1, A_sin_p2,
    B_sin_p1 and B_sin_p2 subfolders. No Drive credentials or raw waveforms
    are embedded in the repository. Extraction preserves acquisition order.
    """
    import io
    manifest=json.loads((DATA/'scalar_ladder_phasors.json').read_text(encoding='utf-8'))
    frequency=local.measured_blocks('nolens')['freqs_hz'].astype(int)
    grouped={}
    for entry in manifest['inputs']:
        path=pathlib.Path(raw_root)/entry['group']/entry['file']
        content=path.read_bytes()
        if hashlib.sha256(content).hexdigest()!=entry['sha256']:
            raise ValueError(f'Raw recording hash mismatch: {path}')
        wave=np.load(io.BytesIO(content),allow_pickle=False)
        if wave.shape!=(6,51200):raise ValueError(f'Unexpected raw shape: {path}')
        phasor=(2*np.fft.rfft(wave,axis=1)/wave.shape[1])[:,frequency].T
        grouped.setdefault(entry['group'],[]).append((entry['file'],phasor))
    arrays={'freqs_hz':frequency}
    for group,values in sorted(grouped.items()):
        values.sort()
        if len(values)!=5:raise ValueError(f'Expected five recordings: {group}')
        arrays[group]=np.stack([p for _,p in values])
        arrays[group+'_files']=np.array([n for n,_ in values])
    np.savez_compressed(DATA/'scalar_ladder_phasors.npz',**arrays)
    print('re-extracted 20 verified recordings',flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if sys.argv[1:2]==['--extract-raw']:
        if len(sys.argv)!=3:raise SystemExit('Use --extract-raw <staging-root>')
        extract_raw(sys.argv[2]);return
    selected=sys.argv[1:] or ['1','2','3','4','5']
    for n in selected:
        globals()['fig'+n]()
    if all((OUT/f'Fig{n}.pdf').exists() for n in range(1,6)):
        write_metadata()


if __name__=='__main__':
    main()
