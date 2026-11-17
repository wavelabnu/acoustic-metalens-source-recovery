"""Numerical and scope checks for the adopted five-figure main."""
from __future__ import annotations
import csv
import json
import re
import numpy as np
from claims import Manuscript, Quoted


def checkpoint_metrics(path, prior, error_covariance, observation, count=32768):
    """Re-evaluate saved weights; do not trust their JSON validation summary.

    Recreate independent validation sources/noise with the recorded seed.
    No training, checkpoint writes, or figure generation occurs here.
    """
    import torch
    from torch import nn
    torch.set_num_threads(2)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    mu=saved['mean'];sd=saved['source_sd'];scale=saved['input_scale']
    np.testing.assert_allclose(mu.numpy(),prior['mean'],rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(sd.numpy()**2,np.diag(prior['covariance']),rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(scale.numpy()**2,
        np.diag(prior['covariance']+error_covariance),rtol=1e-6,atol=1e-7)
    net=nn.Sequential(nn.Linear(2,32),nn.ReLU(),nn.Linear(32,2))
    net.load_state_dict(saved['state_dict']);net.eval()
    rng=torch.Generator().manual_seed(saved['seed']+100000)
    chol=torch.tensor(np.linalg.cholesky(prior['covariance']),dtype=torch.float32)
    noise_chol=torch.tensor(np.linalg.cholesky(error_covariance),dtype=torch.float32)
    coeff=mu+torch.randn(count,2,generator=rng)@chol.T
    observed=coeff+torch.randn(count,2,generator=rng)@noise_chol.T
    with torch.no_grad():
        predicted=(mu+sd*net((observed-mu)/scale)).numpy()
        point=(mu+sd*net((torch.tensor(observation,dtype=torch.float32)-mu)/scale)).numpy()
    gain=np.linalg.solve(prior['covariance']+error_covariance,prior['covariance'].T).T
    exact=prior['mean']+(observed.numpy()-prior['mean'])@gain.T
    return dict(estimate=point,geometry_sha256=saved['geometry_sha256'],
        heldout_rmse_vs_analytic=np.sqrt(np.mean((predicted-exact)**2,axis=0)),
        heldout_coefficient_mse=np.mean((predicted-coeff.numpy())**2,axis=0))


def audit_main(main, led):
    from figure_revision import (model_amplitude_data, measured_amplitude_data,
        measured_amplitude_budget_data, amplitude_tolerance_data, AMPLITUDE_RTOL, prior_data, OUT,
        model_profile_data, measured_profile_data, measured_profile_budget_data, PROFILE_RTOL)
    from professor_v7 import loaders
    import figures as f
    from direct import Config, Lens, stacked_matrix
    from mic_survey import SURVEY
    from resonator_batch import BATCH
    led.header('Adopted main: five figures and current source assumptions')

    def seq(label, values, pattern, doc=main):
        led.check_seq(label, list(values), doc.numbers(pattern))

    def one(label, value, pattern, doc=main):
        led.check(label, value, doc.number(pattern))

    def source_seq(label, values, filename, field, selectors):
        """Check retained figure values even when they are omitted from prose.

        Match exactly one Source Data row per selector and use its printed
        precision, just as for TeX claims. Missing/duplicate rows fail loudly.
        """
        doc = Manuscript('figures/main/current/source_data/' + filename)
        with doc.path.open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        quoted = []
        for selector in selectors:
            matches = [r for r in rows if all(r[k] == v for k, v in selector.items())]
            if len(matches) != 1:
                raise ValueError(f'{filename}: expected one row for {selector}, got {len(matches)}')
            literal = matches[0][field]
            quoted.append(Quoted(literal, doc, re.escape(literal)))
        led.check_seq(label, list(values), quoted)

    def scope(label, start, stop):
        d=Manuscript('main.tex')
        begin=main.text.index(start)
        # An earlier equation can share the closing marker; end this scope locally.
        d.text=main.text[begin:main.text.index(stop,begin+len(start))]
        d.name+=' ('+label+')'
        return d

    def figures_and_protocol():
        paths=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',main.text)
        expected=[f'figures/main/current/Fig{i}.pdf' for i in range(1,6)]
        led.check_ints('exact adopted figures in order',paths,expected)
        led.check_ints('all adopted PDFs exist',[(main.path.parent/p).is_file() for p in paths],[True]*5)
        led.check_ints('phase-ridge Figure 6 removed',
            ['fig:phase' in main.text or 'fig:cal9' in main.text or 'sixteen profiles' in main.text],[False])
        g=loaders.geometry();z=f.measured_blocks('nolens')
        import hashlib
        physical_bytes=(f.EXP_RECON/'run_config_B.json').read_bytes()
        physical=json.loads(physical_bytes)
        geometry=physical['physical_state']['gap_measurement_basis']
        led.check_ints('EXP06 B physical record matches the campaign manifest hash',
            [hashlib.sha256(physical_bytes).hexdigest()],
            ['4cc5b7fa718855b3fe93654afd4cc73b703e12d1f1e5fdab832abf5919009862'])
        led.check_ints('EXP06 B physical record identifies the measured installation',
            [physical['run_id']==str(f.measured_blocks('withlens')['run_id']),
             len(geometry['resonator_positions_mm_centered'])==Config.n_lens,
             np.isclose(np.mean(geometry['resonator_positions_mm_centered']),0),
             np.isclose(np.mean(geometry['speaker_positions_mm_centered']),0)],
            [True]*4)
        one('mounted lens pitch mm',geometry['resonator_pitch_mm'],
            r'ten resonators are spaced at \$NUM\$~mm')
        gaps=geometry['raw_resonator_gaps_mm_left_to_right']
        seq('recorded throat-to-speaker-line gap range mm',[min(gaps),max(gaps)],
            r'throat-to-speaker-line gap readings span \$NUM\$--\$NUM\$~mm')
        one('source pitch mm',f.BENCH_D*1000,r'loudspeakers at \$NUM\$~mm pitch')
        seq('bench frequency wavelength and pitch',
            [f.BENCH_F0,f.BENCH_C/f.BENCH_F0*1000,f.BENCH_D/(f.BENCH_C/f.BENCH_F0)],
            r'At \$NUM\$~Hz, \$\\lambda_0\\simeqNUM\$~mm, so the pitch is \$NUM\\lambda_0\$')
        seq('aperture convention and physical centre span',[8,8*f.BENCH_D*1000,7,7*f.BENCH_D*1000],
            r'\$L=NUMd=NUM\$~mm; the distance between the outer source centres is \$NUMd=NUM\$~mm')
        seq('microphone geometry',[SURVEY.arc_m,SURVEY.outlier[1],SURVEY.span_deg],
            r'approximately \$NUM\$~m and one at \$NUM\$~m, over an angular span of \$NUM\^\\circ\$')
        seq('recording tones and band',[len(z['freqs_hz']),z['freqs_hz'].min(),z['freqs_hz'].max()],
            r'stack \$NUM\$ tones over \$NUM\$--\$NUM\$~Hz')
        seq('batch resonance summary',[BATCH.f0_hz.min(),BATCH.f0_hz.max(),BATCH.f0_hz.mean()],
            r'frequencies span \$NUM\$--\$NUM\$~Hz with mean \$NUM\$~Hz')
        one('electrical drive setting',float(z['drive_level_v']),r'nominal \$NUM\$~V drive setting')
        one('reference sound speed',f.BENCH_C,r'sound speed \$c=NUM\$~m')
        one('retained separate image shots',sum(str(k).startswith('image:') for k in z['kinds']),
            r'contains \$NUM\$ separate image')
        seq('DFT wavenumbers',2*np.pi*np.arange(1,5)/(8*f.BENCH_D),
            r'wavenumbers are approximately \$NUM\$, \$NUM\$, \$NUM\$ and \$NUM\$~rad')
        led.check_ints('acquisition repeats and matched drives',
            [int(z['n_cal_reps']),len(f.measured_pattern_samples('staggered','nolens')[0]),
             all(np.array_equal(z['D'],f.measured_blocks(t)['D']) for t in ('withlens','nolens_b6'))],
            [9,5,True])

    def pair():
        # Frozen illustrative fields retain the original producer and noise.
        # The pair export holds exact separation; Figure 1 fields hold its
        # common-noise convention. Check against the current package below.
        cfg=Config();T=[]
        for wl in (False,True):
            T.append(stacked_matrix(cfg,cfg.freqs(20),Lens(cfg),with_lens=wl,src_scatters=True))
        vh=np.linalg.svd(np.vstack([T[0].real,T[0].imag]),full_matrices=False)[2]
        s1=(vh[0]+vh[-1])/np.sqrt(2);s2=(vh[0]-vh[-1])/np.sqrt(2)
        sigma=np.linalg.svd(T[0],compute_uv=False)[0]*10**(-64/20)
        ds=[np.sqrt(2)*np.linalg.norm(t@(s2-s1))/sigma for t in T]
        seq('model candidate discriminant separation',ds,
            'is only \\$NUM\\$ without the lens, compared with \\$NUM\\$ with it')
        powers=[np.linalg.norm(t@source)**2/(t.shape[0]*sigma*sigma)
                for t in T for source in (s1,s2)]
        led.check_ints('both model candidates exceed total noise in both arms',
            [all(power>1 for power in powers)],[True])
        led.check_ints('orthogonal equal-drive-norm candidates',
            [np.isclose(s1@s2,0),np.isclose(s1@s1,s2@s2)],[True,True])
        seq('caption separation at displayed precision',ds,
            r'separation is \$d=NUM\$ without the lens and \$NUM\$ with it')
        from direct.physics import transfer_matrix,field_on_grid
        fields=loaders.fig01_pair_fields();frozen=loaders.fig01_pair()['model']['pairs']['rule_v1_v8']
        source1=np.asarray(frozen['s1']);source2=np.asarray(frozen['s2'])
        extrema=np.vstack([(source1+source2)/np.sqrt(2),(source1-source2)/np.sqrt(2)])
        led.check_ints('actual Figure 1 source glyphs are the stated real-SVD extreme pair',
            [np.allclose(abs(extrema@vh[[0,-1]].T),np.eye(2),atol=1e-10)],[True])
        x,y=np.meshgrid(fields['grid_x'],fields['grid_y'])
        tone=float(fields['freqs_over_f0'][int(fields['t_b'])]);noise=float(fields['sigma_n_nolens'])
        mic_matches=[];map_matches=[];inset_matches=[]
        for arm,wl in [('nolens',False),('lens',True)]:
            microphones=np.array([transfer_matrix(cfg,float(freq),Lens(cfg),with_lens=wl,
                src_resonant=True,src_scatters=True)@source1 for freq in fields['freqs_over_f0']])
            p1=field_on_grid(cfg,tone,x,y,Lens(cfg),with_lens=wl,src_resonant=True,drive=source1)
            p2=field_on_grid(cfg,tone,x,y,Lens(cfg),with_lens=wl,src_resonant=True,drive=source2)
            mic_matches.append(np.allclose(microphones,fields['mic_'+arm+'_p_s1'],rtol=1e-11,atol=1e-13))
            map_matches.append(np.allclose((p1-p2).real/noise,
                fields['field_'+arm+'_tb_delta_total_re_over_sigma'],rtol=1e-11,atol=1e-10))
            inset_matches.append(np.allclose(p1.real/noise,
                fields['field_'+arm+'_tb_s1_total_re_over_sigma'],rtol=1e-11,atol=1e-10))
        led.check_ints('Figure 1 microphone fields reproduce the full frozen frequency stack',mic_matches,[True,True])
        led.check_ints('Figure 1 complete plotted field grids reproduce package physics',map_matches,[True,True])
        led.check_ints('Figure 1 complete inset fields reproduce package physics',inset_matches,[True,True])

    def model():
        state,no=model_amplitude_data();modes=list(no)
        z=loaders.npz('fig02_model_thresholds.npz');order=list(map(str,z['modes']))
        indices=[order.index(m) for m in modes]
        source_seq('Figure 2 displayed power margins',z['nolens_margin_ref'][indices],
            'fig02_a_detection.csv','margin_db_re_floor',[dict(mode=m) for m in modes])
        source_seq('Figure 2 prior-free scalar RMS values',
            [state['comparison'][tag][m]['rms'] for tag in ('nolens','withlens') for m in modes],
            'fig02_b_estimation.csv','scalar_complex_rms',
            [dict(arm=tag,mode=m,draw='0') for tag in ('nolens','withlens') for m in modes])
        one('selected model scalar RMS percentage',100*state['comparison']['nolens']['cos_p4']['rms'],
            'finest pattern.*?RMS amplitude error of only \\$NUM\\\\%\\$')
        T=state['operator'];u,s,vh=np.linalg.svd(T,full_matrices=False)
        rank=int(np.count_nonzero(s>float(z['rank_edge'])*state['sigma']))
        projected=[abs(np.vdot(no[m]['truth'],vh[:rank].conj().T@(vh[:rank]@no[m]['truth']))) for m in modes]
        phi=np.asarray(z['phi']);expected=(vh[:rank].conj().T@(vh[:rank]@phi.T)).T
        led.check_ints('all eight frozen Figure 2 noise-free profiles reproduce the current operator',
            [np.allclose(expected,z['nolens_return_noise_free'],rtol=1e-10,atol=1e-12)],[True])
        profiles=model_profile_data();rms=[];consistent=[]
        for arm,key in (('nolens','nolens'),('withlens','lens')):
            A=state['operators'][arm];u,sv,vh=np.linalg.svd(A,full_matrices=False)
            r=profiles[arm][modes[0]]['rank'];inverse=(vh[:r].conj().T/sv[:r])@u[:,:r].conj().T
            for m in modes:
                phi=profiles[arm][m]['truth'];i=order.index(m)
                independent=np.sqrt(np.linalg.norm(inverse@A@phi-phi)**2
                                    +state['sigma']**2*np.linalg.norm(inverse)**2)
                rms.append(independent)
                consistent.append(np.isclose(independent,z[key+'_err_ref_analytic_rms'][i]))
                source_seq('Figure 2 preserved first display '+arm+'/'+m,
                    profiles[arm][m]['estimates'][0].real,'fig02_c_recovery.csv','estimate_re',
                    [dict(arm=arm,mode=m,speaker=str(j)) for j in range(1,9)])
                source_seq('Figure 2 preserved first imaginary components '+arm+'/'+m,
                    profiles[arm][m]['estimates'][0].imag,'fig02_c_recovery.csv','estimate_im',
                    [dict(arm=arm,mode=m,speaker=str(j)) for j in range(1,9)])
        led.check_ints('full inverse bias and noise trace reproduce frozen model RMS',consistent,[True]*6)
        source_seq('Figure 2 common full-profile RMS scores',rms,'fig02_c_recovery.csv','profile_rms',
            [dict(arm=arm,mode=m,speaker='1') for arm in ('nolens','withlens') for m in modes])
        seq('selected model RMS profile errors',[rms[2],rms[5]],
            'RMS relative profile error is \\$NUM\\$\\..*?latter to \\$NUM\\$')
        led.check_ints('one half-mean-square profile tolerance in main and driver',
            [np.isclose(PROFILE_RTOL**2,.5),r'E_{\rm prof}\le1/\sqrt2' in main.text],[True,True])
        one('reference model budget',state['budget_db'],'Figure~\\\\ref\\{fig:model\\} uses \\$NUM\\$~dB')
        one('model noise-to-gain ratio mantissa',state['sigma']*1e4,
            r'noise-to-gain ratio \$\\sigma_n/\\sigma_1=10\^\{-64/20\}\\simeq\s*NUM\\times10\^\{-4\}\$')
        for mode in modes:
            response=(T@no[mode]['truth']).reshape(20,6)
            mic_power=10*np.log10(np.sum(abs(response)**2,axis=0)/(20*state['sigma']**2))
            source_seq('Figure 2 per-microphone noise normalization '+mode,mic_power,
                'fig02_a_detection.csv',mode+'_power_db_re_floor',
                [dict(microphone=str(j)) for j in range(1,7)])
        one('model resonant self-term coefficient',Config.q_factor,r'parameters are \$Q=NUM\$')
        one('assumed lens absorption',Config.rw_loss,r'absorption \$\\gamma=NUM\$')
        one('assumed source resonance',Config.f_res_src_frac,r'source resonance \$NUMf_0\$')
        cfg=Config();lens=Lens(cfg)
        seq('model lens transverse endpoints',
            [lens.YL.min()/cfg.lam_res,lens.YL.max()/cfg.lam_res],
            r'from \$y=NUM\\lambda_0\$ to \$y=NUM\\lambda_0\$')
        led.check_ints('model arrays share a transverse centre and uniform lens pitch',
            [np.isclose(cfg.y_src.mean(),0),np.isclose(lens.YL.mean(),0),
             np.allclose(np.diff(lens.YL),2*cfg.d/3)],[True,True,True])
        led.check_ints('scalar RMS passes stated target',
            [all(v['rms']<.04 for arm in state['comparison'].values() for v in arm.values())],[True])
        one('unit-scale mean-square precision target',.04**2,
            r'require \$\\mathbb E\|\\hat A-A\|\^2/\|A\|\^2\\leNUM\$')
        seq('scalar-fit precision signal-to-noise ratio',[1/.04**2,10*np.log10(1/.04**2)],
            r'must be at least \$NUM\$ \(\$NUM\$~dB\)')
        choice=amplitude_tolerance_data()
        seq('empirical amplitude tolerance cohort',[100*choice['coverage'],choice['count'],choice['recording_count']],
            r'empirical \$NUM\$th percentile of five-repeat RMS scalar errors across \$NUM\$ pattern--session cases \(\$NUM\$ recordings')
        seq('empirical percentile and rounded amplitude tolerance',[100*choice['cutoff'],100*choice['selected']],
            r'The percentile is \$NUM\\%\$; rounding upward to the next whole percentage point gives \$NUM\\%\$')
        led.check_ints('smallest whole-percent tolerance covers the selected case fraction',
            [np.isclose(AMPLITUDE_RTOL,choice['selected']),
             np.count_nonzero(choice['case_rms']<=AMPLITUDE_RTOL)>=choice['coverage']*choice['count'],
             np.count_nonzero(choice['case_rms']<=AMPLITUDE_RTOL-choice['step'])<choice['coverage']*choice['count']],
            [True,True,True])

    def measured_budget():
        C=loaders.json_('fig01_channels.json');s=[];counts=[]
        detection={}
        for tag in ('nolens','withlens','nolens_b6'):
            a=f.measured_blocks(tag);cal=np.flatnonzero(a['kinds']=='cal')
            H=a['Y'][cal].transpose(1,2,0);sigma=np.std(a['Y_baseline_all'],axis=0,ddof=1)
            sv=np.linalg.svd((H/sigma[:,:,None]).reshape(-1,8),compute_uv=False)
            s.append(sv);counts.append(int(np.count_nonzero(sv>C['edge'])))
            patterns=loaders.returned_profiles('B',4)[0]
            patterns=patterns/np.linalg.norm(patterns,axis=1)[:,None]
            detection[tag]=20*np.log10(sv[0]*np.linalg.norm(sigma)
                /np.linalg.norm(H.reshape(-1,8)@patterns.T,axis=0))
        one('measured complex cutoff',3*(np.sqrt(41*6)+np.sqrt(8)),r'operational cutoff \$3\(\\sqrt\{KM\}\+\\sqrt N\)=NUM\$')
        seq('selected weak and strongest measured gains',[s[0][3],s[1][3],s[0][0],s[1][0]],
            r'fourth singular value (?:rises )?from \$NUM\$ to \$NUM\$,.*?strongest(?: value)? rises from \$NUM\$ to \$NUM\$')
        seq('Supplement baseline-weighted gains use the same convention as main',
            [s[0][3],s[1][3],s[0][0],s[1][0]],
            r'fourth singular value (?:rises )?from \$NUM\$ to \$NUM\$,.*?strongest(?: value)? rises from \$NUM\$ to \$NUM\$',
            Manuscript('supplement.tex'))
        for arm,sv in [('no_lens',s[0]),('with_lens',s[1])]:
            source_seq(arm+' fifth gain in Figure 3 Source Data',[sv[4]],
                'fig03_d_channels.csv',arm+'_singular_value',[dict(channel='5')])
        seq('available baseline-normalized budgets',[20*np.log10(s[0][0]),20*np.log10(s[1][0])],
            r'budgets, \$NUM\$ and \$NUM\$~dB')
        z=loaders.npz('fig03_measured_budget.npz');modes=list(map(str,z['modes']));idx=[modes.index(m) for m in ('cos_p2','sin_p2')]
        fine,coarse=[modes.index(m) for m in ('cos_p3','cos_p2')]
        seq('finer-pattern recorded response levels',
            [z[k+'_margin_shot_median'][fine] for k in ('nolens','lens')],
            'finer \\$\\\\lambda/19.5\\$ cosine gives recorded power levels of \\$NUM\\$~dB without the lens and \\$NUM\\$~dB with it')
        seq('coarser-pattern recorded response levels',
            [z[k+'_margin_shot_median'][coarse] for k in ('nolens','lens')],
            r'coarser \$\\lambda/13\$ cosine gives \$NUM\$ and \$NUM\$~dB, respectively')
        drawn=loaders.returned_profiles('B',4)[0]
        led.check_ints('response-order comparison has equal drive norms and reverses',
            [np.isclose(np.linalg.norm(drawn[fine]),np.linalg.norm(drawn[coarse])),
             z['nolens_margin_shot_median'][fine]<z['nolens_margin_shot_median'][coarse],
             z['lens_margin_shot_median'][fine]>z['lens_margin_shot_median'][coarse]],
            [True,True,True])
        for arm,field in [('nolens','no_lens_shots_median_db_re_floor'),
                          ('lens','with_lens_shots_median_db_re_floor')]:
            source_seq(arm+' recorded Figure 3 power levels',z[arm+'_margin_shot_median'],
                'fig03_a_responses.csv',field,[dict(mode=m) for m in modes])
        # A full-profile crossing may be set by a discrete TSVD rank
        # change; check that interpretation against every plotted threshold.
        coincident=0
        recovery=measured_profile_budget_data()
        led.check_ints('budget grid reaches full source rank in both arms',
            [int(recovery[a]['ranks'][-1]) for a in ('nolens','lens')],[8,8])
        for arm,sv in [('nolens',s[0]),('lens',s[1])]:
            noise=sv[0]*10**(-z['budgets']/20)
            entering=[z['budgets'][np.flatnonzero(v>float(z['rank_edge'])*noise)[0]]
                      for v in sv]
            coincident+=sum(np.any(np.isclose(b,entering,rtol=0,atol=1e-9))
                            for b in recovery[arm]['budget_db'])
        one('profile-error thresholds coincide with floor-rank entries',coincident,
            'all \\$NUM\\$ plotted recovery thresholds coincide, on the sampled budget grid,')
        led.check_ints('raw-power detection budgets reproduce the adopted grid',
            [np.allclose(np.ceil(detection[tag]*10)/10,z[arm+'_B_detected'],atol=1e-9)
             for tag,arm in [('nolens','nolens'),('withlens','lens')]],[True,True])
        amplitude=measured_amplitude_budget_data()
        seq('selected reconstruction-budget savings',
            (recovery['nolens']['budget_db']-recovery['lens']['budget_db'])[idx],
            r'saving is \$NUM\$~dB for the cosine and \$NUM\$~dB for the sine')
        adverse=modes.index('sin_p3')
        seq('adopted criterion includes a lens budget increase',
            [recovery[arm]['budget_db'][adverse] for arm in ('nolens','lens')],
            r'For the \$\\lambda/19.5\$ sine, the required budget instead rises from \$NUM\$ to \$NUM\$~dB with the lens')
        seq('profile-error passing counts at each available budget',
            [np.count_nonzero(recovery[arm]['rms_at_available']<=PROFILE_RTOL)
             for arm,sv in (('nolens',s[0]),('lens',s[1]))],
            'For profile recovery, \\$NUM\\$ of eight patterns pass without the lens and \\$NUM\\$ with it')
        with (OUT/'source_data/fig03_bcd_budgets.csv').open(encoding='utf-8') as stream:
            rows=list(csv.DictReader(stream))
        led.check_ints('Figure 3 known-pattern amplitude Source Data',
            [len(rows),all(np.isclose(float(row['B_estimated_db']),amplitude['nolens' if row['arm']=='no lens' else 'lens'][row['mode']]['budget_db']) for row in rows),
             'r_at_available' not in rows[0]],[16,True,True])
        seq('measured patterns passing scalar amplitude estimation',
            [sum(row['rms_at_available']<=row['amplitude_rtol'] for row in amplitude[arm].values()) for arm in ('nolens','lens')],
            r'At the acquired background level, \$NUM\$ of eight patterns pass.*?without the lens and all \$NUM\$ pass with it')
        seq('independent cutoff conventions',[*map(f.measured_rank,('nolens','withlens','nolens_b6')),*counts],
            r'counts \$NUM/NUM/NUM\$ for the initial no-lens, with-lens and repeat no-lens measurements, respectively,.*?gives \$NUM/NUM/NUM\$')
        source_seq('unit-pattern margins retained in Figure 3 Source Data',
            [z[k+'_margin_ref'][i] for k in ('nolens','lens') for i in range(len(modes))],
            'fig03_bcd_budgets.csv','margin_unit_norm_db',
            [dict(arm=arm,mode=m) for arm in ('no lens','with lens') for m in modes])
        source_seq('all Figure 3 detection Source Data',
            [z[k+'_B_detected'][i] for k in ('nolens','lens') for i in range(len(modes))],
            'fig03_bcd_budgets.csv','B_detected_db',
            [dict(arm=arm,mode=m) for arm in ('no lens','with lens') for m in modes])
        source_seq('all Figure 3 full-profile budget Source Data',
            [recovery[k]['budget_db'][i] for k in ('nolens','lens') for i in range(len(modes))],
            'fig03_bcd_budgets.csv','B_tsvd_profile_rms_db',
            [dict(arm=arm,mode=m) for arm in ('no lens','with lens') for m in modes])
        source_seq('all Figure 3 available profile RMS Source Data',
            [recovery[k]['rms_at_available'][i] for k in ('nolens','lens') for i in range(len(modes))],
            'fig03_bcd_budgets.csv','profile_rms_at_available',
            [dict(arm=arm,mode=m) for arm in ('no lens','with lens') for m in modes])
        # Independent risk calculation: inverse bias plus the complete noise trace.
        independent=[]
        for arm in ('nolens','lens'):
            row=recovery[arm];A=row['operator'];u,sv,vh=np.linalg.svd(A,full_matrices=False)
            r=row['rank_at_available'];inverse=(vh[:r].conj().T/sv[:r])@u[:,:r].conj().T
            for i,phi in enumerate(row['patterns']):
                independent.append(np.isclose(row['rms_at_available'][i],
                    np.sqrt(np.linalg.norm(inverse@A@phi-phi)**2+np.linalg.norm(inverse)**2)))
        led.check_ints('all measured-operator profile RMS scores reproduce full inverse risk',independent,[True]*16)
        seq('main empirical error-scale provenance',
            [100*f.measured_prediction_error(t)['median'] for t in ('nolens','withlens','nolens_b6')],
            r'Median recording prediction errors are \$NUM\\%\$, \$NUM\\%\$ and \$NUM\\%\$ for the initial no-lens, with-lens and repeat no-lens measurements')

    def threshold_sensitivity():
        from figure_revision import criterion_sensitivity_data
        result=criterion_sensitivity_data()
        supp=Manuscript('supplement.tex')
        shifts=result['amplitude_shift_db']
        seq('scalar tolerance budget shifts',[shifts[.02],shifts[.08]],
            r'by \$NUM\$ or \$NUM\$~dB, respectively',supp)
        choice=amplitude_tolerance_data()
        seq('empirical case RMS rank and achieved coverage',[choice['rank'],100*choice['cutoff'],100*choice['selected'],choice['passing_count'],choice['count']],
            r'the \$NUM\$th ordered RMS error is \$NUM\\%\$.*?selects \$NUM\\%\$, which covers \$NUM\$ of \$NUM\$ cases',supp)
        sensitivity_counts=[sum(row['rms_at_available']<=tau for row in measured_amplitude_budget_data(tau)[arm].values())
                            for tau in (.02,.08) for arm in ('nolens','lens')]
        seq('measured scalar passing counts at sensitivity tolerances',sensitivity_counts,
            r'gives \$NUM\$/\$NUM\$ passing patterns without/with the lens; relaxing it to \$8\\%\$ gives \$NUM\$/\$NUM\$',supp)
        targets=(.5,1/np.sqrt(2),.9)
        savings={t:result['arms']['nolens']['profile_budget_db'][t]
                   -result['arms']['lens']['profile_budget_db'][t] for t in targets}
        modes=loaders.mode_order()
        seq('profile-RMS sensitivity cosine/sine',
            [savings[t][modes.index(m)] for m in ('cos_p2','sin_p2') for t in targets],
            r'give lens budget savings of \$NUM\$, \$NUM\$ and \$NUM\$~dB for the.*?cosine, and \$NUM\$, \$NUM\$ and \$NUM\$~dB for the corresponding sine',supp)
        seq('profile-RMS sensitivity finest cosine',
            [savings[t][modes.index('cos_p4')] for t in targets],
            r'For the \$\\lambda/26\$ cosine, the savings are \$NUM\$, \$NUM\$ and \$NUM\$~dB',supp)
        seq('main explains the finest-pattern target sensitivity',
            [savings[t][modes.index('cos_p4')] for t in (1/np.sqrt(2),.5)],
            r'tolerance from \$1/\\sqrt2\$ to \$0.5\$ changes the calculated budget saving from \$NUM\$ to \$NUM\$~dB')
        led.check_ints('model and with-lens ladder RMS errors pass 2 percent',
            [all(v['rms']<.02 for row in model_amplitude_data()[0]['comparison'].values() for v in row.values()),
             np.all(result['arms']['lens']['amplitude_rms']<.02)],[True,True])
        seq('floor multiplier channel-entry shifts',20*np.log10(np.array([2.,4.])/3),
            r'channel-entry budgets by \$NUM\$ or \$NUM\$~dB',supp)
        base=result['arms']['nolens']['floor_factor_budget_db'][3.]-result['arms']['lens']['floor_factor_budget_db'][3.]
        change=max(np.max(abs(result['arms']['nolens']['floor_factor_budget_db'][c]
                              -result['arms']['lens']['floor_factor_budget_db'][c]-base))
                   for c in (2.,4.))
        one('floor multiplier changes paired savings by at most one grid step',change,
            r'every lens--no-lens saving by at most \$NUM\$~dB',supp)

    def parameter_choices():
        from figure_revision import model_parameter_choice_data,recorded_setting_sensitivity_data
        p=model_parameter_choice_data();supp=Manuscript('supplement.tex')
        seq('interior model reference-budget interval',p['budget_interval_db'],
            r'rank is three throughout \$NUM\$--\$NUM\$~dB',supp)
        # The submitted text adds the narrower interval preserving all six
        # Figure 2c pass/fail outcomes. Recompute complete risk at every grid
        # point, with source bias and propagated inverse noise kept separate.
        state,_=model_amplitude_data()
        budgets=np.arange(round(p['budget_interval_db'][0]*10),
                          round(p['budget_interval_db'][1]*10)+1)/10
        stable=np.ones(len(budgets),dtype=bool)
        ranks={}
        for arm in ('nolens','withlens'):
            T=state['operators'][arm]
            _,sv,vh=np.linalg.svd(T,full_matrices=False)
            patterns=np.array([row['truth'] for row in state['comparison'][arm].values()])
            norm2=np.sum(abs(patterns)**2,axis=1)
            sigma=sv[0]*10**(-budgets/20)
            edge=3*(np.sqrt(T.shape[0])+np.sqrt(T.shape[1]))
            keep=sv[:,None]>edge*sigma[None,:]
            captured=abs(vh@patterns.T)**2
            bias=np.maximum(norm2[:,None]-captured.T@keep,0)
            variance=sigma**2*np.sum(keep/sv[:,None]**2,axis=0)
            risk=np.sqrt((bias+variance)/norm2[:,None])
            reference=np.flatnonzero(np.isclose(budgets,state['budget_db']))[0]
            stable&=np.all((risk<=PROFILE_RTOL)==(risk[:,reference]<=PROFILE_RTOL)[:,None],axis=0)
            ranks[arm]=keep.sum(axis=0)
        kept=budgets[stable]
        assert len(kept) and np.allclose(np.diff(kept),.1)
        seq('submitted main unchanged model-profile verdict interval',[kept[0],kept[-1]],
            r'unchanged throughout the sampled \$NUM\$--\$NUM\$~dB range')
        seq('Supplement unchanged model-profile verdict interval',[kept[0],kept[-1]],
            r'narrower interval \$NUM\$--\$NUM\$~dB',supp)
        one('Supplement number of unchanged-verdict model budgets',len(kept),
            r'Across these \$NUM\$ budgets',supp)
        words=re.search(r'retained ranks are (\w+) and (\w+)',supp.text).groups()
        integers={word:i for i,word in enumerate('zero one two three four five six seven eight nine ten'.split())}
        led.check_ints('Supplement retained ranks throughout unchanged-verdict interval',
            [int(np.unique(ranks[arm][stable]).item()) for arm in ('nolens','withlens')],
            [integers[word] for word in words])
        seq('isolated linewidth and absorption share',[p['linewidth_pct'],p['loss_share_pct']],
            r'fractional linewidth is \$NUM\\%\$; absorption contributes \$NUM\\%\$ of the total isolated damping',supp)
        one('model tone-grid step',p['tone_step_pct'],r'tone spacing of \$NUM\\%\$ of \$f_0\$',supp)
        one('source reactive-to-damping ceiling',p['driver_reactive_pct'],
            r'at most \$NUM\\%\$ of the damping term',supp)
        one('model microphone angular half-aperture',p['microphone_half_angle_deg'],
            r'angular half-aperture of approximately \$NUM\^\\circ\$',supp)
        seq('lens absorption control floor counts',[r['floor_rank'] for r in p['loss_sweep']],
            r'complex floor counts are \$NUM/NUM/NUM\$',supp)
        seq('lens absorption control weakest margins',[r['weakest_margin'] for r in p['loss_sweep']],
            r'weakest-gain margins.*?are \$NUM\$, \$NUM\$ and \$NUM\$',supp)
        led.check_ints('lens absorption control retains eight at noise-level cut',
            [r['noise_rank'] for r in p['loss_sweep']],[8]*3)
        r=recorded_setting_sensitivity_data()
        led.check_ints('all 810 rank/weight/repeat returns independently reproduce frozen controls',
            [r['max_frozen_difference']<1e-10],[True])
        seq('coarsest recorded pattern rank/weight sensitivity',
            [r['improvement_counts'][k][0] for k in (4,5,6)],
            r'for \$NUM\$, \$NUM\$ and \$NUM\$ of the nine weight choices',supp)
        led.check_ints('two finer recorded profiles improve across all tested weights and ranks',
            [r['improvement_counts'][k][i] for k in (4,5,6) for i in (1,2)],[9]*6)

    def recorded():
        fits=measured_amplitude_data();no=fits['A']['cos_p4'];yes=fits['B']['cos_p4']
        d=scope('recorded Results',r'\subsection{The lens improves',r'\subsection{Source assumptions')
        seq('recorded scalar-error medians',[100*np.median(no['errors']),100*np.median(yes['errors'])],
            'error of the known-profile complex amplitude fit falls from \\$NUM\\\\%\\$ without the lens to \\$NUM\\\\%\\$ with it',d)
        seq('conditional scalar limits',[100*no['conditional_limit'],100*yes['conditional_limit']],
            r'conditional limit changes from \$NUM\\%\$ to \$NUM\\%\$',main)
        c=loaders.fig04_tsvd_complex();errors=[];weighted_differences=[]
        for arm in ('A','B'):
            block=f.measured_blocks('nolens' if arm=='A' else 'withlens')
            H=block['Y'][block['kinds']=='cal'].transpose(1,2,0).reshape(-1,8)
            weights=1/abs(block['Y_baseline_all'][0]).reshape(-1)
            u,s,vh=np.linalg.svd(weights[:,None]*H,full_matrices=False)
            for m in ('sin_p1','sin_p2','cos_p4'):
                truth=c['v_'+m];estimate=c[f'ret45_{arm}_{m}_rank5'][0,0]
                errors.append(np.linalg.norm(estimate-truth)/np.linalg.norm(truth))
                y=fits[arm][m]['recordings'][0].reshape(-1)
                rebuilt=vh[:5].conj().T@((u[:,:5].conj().T@(weights*y))/s[:5])
                weighted_differences.append(np.max(abs(rebuilt-estimate)))
        led.check_ints('Figure 4 weighted TSVD reproduces all six frozen returns',
            [max(weighted_differences)<1e-10],[True])
        source_seq('Figure 4 displayed single-record complex errors',errors,
            'fig04_source_profiles.csv','full_relative_error',
            [dict(arm=arm,mode=m,speaker='1',record='0') for arm in ('A','B')
             for m in ('sin_p1','sin_p2','cos_p4')])
        profiles=measured_profile_data();rms=[profiles[a][m]['rms'] for a in ('A','B')
                                           for m in ('sin_p1','sin_p2','cos_p4')]
        source_seq('Figure 4 five-record full-profile RMS scores',rms,
            'fig04_source_profiles.csv','profile_rms',
            [dict(arm=a,mode=m,speaker='1',record='0') for a in ('A','B')
             for m in ('sin_p1','sin_p2','cos_p4')])
        seq('selected recorded RMS profile errors',[rms[2],rms[5]],
            'full-profile RMS relative error falls from \\$NUM\\$ to \\$NUM\\$',d)
        seq('recorded full-profile passing counts',
            [sum(v['rms']<=PROFILE_RTOL for v in profiles[a].values()) for a in ('A','B')],
            r'\$NUM/3\$ no-lens cases and \$NUM/3\$ with-lens cases pass',d)
        with (OUT/'source_data/fig04_source_profiles.csv').open(encoding='utf-8') as stream:
            exported=list(csv.DictReader(stream))
        all_records=[];scores=[]
        for arm in ('A','B'):
            for mode in ('sin_p1','sin_p2','cos_p4'):
                truth=c['v_'+mode];returns=[]
                for record in range(5):
                    rows=sorted((r for r in exported if r['arm']==arm and r['mode']==mode
                                 and r['record']==str(record)),key=lambda r:int(r['speaker']))
                    estimate=np.array([complex(float(r['estimate_re']),float(r['estimate_im'])) for r in rows])
                    all_records.append(len(rows)==8 and np.allclose(estimate,
                        c[f'ret45_{arm}_{mode}_rank5'][0,record],atol=6e-6,rtol=6e-6))
                    returns.append(estimate)
                scores.append(np.isclose(np.sqrt(np.mean(np.sum(abs(np.array(returns)-truth)**2,axis=1)))
                                         /np.linalg.norm(truth),profiles[arm][mode]['rms'],atol=3e-6))
        led.check_ints('all thirty measured profiles retained in RMS Source Data',all_records,[True]*30)
        led.check_ints('six five-record RMS scores reproduce exported full complex profiles',scores,[True]*6)
        one('recording bound scale',f.MEAS_REL_FLOOR,r'uses \$\\epsilon=NUM\\sigma_1')
        seq('observed scalar errors passing target',
            [sum(np.count_nonzero(v['errors']<=.04) for v in fits[arm].values()) for arm in ('A','B')],
            r'\$NUM\$ of the fifteen no-lens scalar errors and all \$NUM\$ with-lens errors pass')

    def priors():
        data=prior_data();d=scope('prior Results',r'\subsection{Source assumptions',r'\section{Discussion}')
        patterns=[r'errors are \$NUM\\%\$, \$NUM\\%\$ and \$NUM\\%\$ without the lens\. With the lens, the errors are \$NUM\\%\$, \$NUM\\%\$ and \$NUM\\%\$',
            'estimates are \\$NUM\\\\%\\$, \\$NUM\\\\%\\$ and \\$NUM\\\\%\\$, respectively\\. With the lens, the corresponding errors are \\$NUM\\\\%\\$, \\$NUM\\\\%\\$ and \\$NUM\\\\%\\$']
        for group,pat in zip(('model','measured'),patterns):
            errors=[];shifts=[]
            for tag in ('nolens','withlens'):
                row=data[group][tag];truth=row['profiles']['truth'];values=[]
                for key in ('none','correct','wrong'):
                    values.append(100*np.linalg.norm(row['profiles'][key]-truth)/np.linalg.norm(truth))
                errors.extend(values);shifts.append(values[1]-values[0])
            if group=='measured':
                seq(group+' prior full-profile errors',errors,pat,d)
            source_seq(group+' full-profile errors in Figure 5 Source Data',np.asarray(errors)/100,
                'fig05_prior_comparison.csv','shown_profile_relative_error',
                [dict(group=group,arm=arm,estimator=key) for arm in ('nolens','withlens')
                 for key in ('none','correct','wrong')])
            source_seq(group+' prior changes in Figure 5 Source Data',shifts,
                'fig05_prior_shifts.csv','error_change_percentage_points',
                [dict(group=group,arm=arm,prior='correct') for arm in ('nolens','withlens')])
            all_errors=np.asarray(errors).reshape(2,3)
            source_seq(group+' wrong-prior changes from the data-only estimate',all_errors[:,2]-all_errors[:,0],
                'fig05_prior_shifts.csv','error_change_percentage_points',
                [dict(group=group,arm=arm,prior='wrong') for arm in ('nolens','withlens')])
            led.check_ints(group+' physical basis fixed across operators',
                [np.array_equal(data[group]['nolens']['basis'],data[group]['withlens']['basis'])],[True])
        row=data['measured']['nolens']
        led.check_ints('all target records held out from bias/covariance',
            [len(row['record_indices']),len(row['training_classes']),bool(set(row['record_indices'])&set(row['training_indices']))],
            [5,17,False])
        one('model prior budget',24,r'priors at \$NUM\$~dB relative budget')
        seq('model prior correlations',[.9,-.9],r'correlations \$\+NUM\$ or \$NUM\$')
        one('soft-prior background variance',row['priors']['correct']['drive_variance'][1],r'and \$NUM\$ elsewhere')
        led.check_ints('Gaussian update uses covariance of source prior',
            [r'\bm\mu+C(C+S)^{-1}(\mathbf z-\bm\mu)' in main.text],[True])
        uncertainty=scope('Figure 5 uncertainty',r'\label{eq:uncertainty}',r'\end{equation}')
        one('Figure 5 squared Mahalanobis radius',4,r'\\le\s*NUM',uncertainty)
        one('two-dimensional Gaussian probability inside radius two',
            100*(-np.expm1(-2)),'true displayed coefficients in \\$NUM\\\\%\\$ of repeated realizations',d)
        from direct.mode_prior import operator,real_operator
        from figure_revision import ellipse
        import matplotlib.pyplot as plt
        consistent=[]
        for tag in ('nolens','withlens'):
            row=data['model'][tag];design=real_operator(operator(tag=='withlens')@row['basis'])
            expected=((10**(-24/20))**2/2)*np.linalg.inv(design.T@design)
            consistent.append(np.allclose(row['data_cov'],expected,rtol=1e-10,atol=1e-12))
        led.check_ints('Figure 5 model covariance matches real Gaussian least squares',consistent,[True,True])
        consistent=[]
        for group,arms in data.items():
            for row in arms.values():
                fig,ax=plt.subplots()
                ellipse(ax,row['estimates']['none'],row['data_cov'],'black')
                patch=ax.patches[-1];points=patch.get_patch_transform().transform(
                    np.c_[np.cos(np.arange(16)*np.pi/8),np.sin(np.arange(16)*np.pi/8)])
                delta=points-row['estimates']['none']
                radii=np.einsum('ni,ij,nj->n',delta,np.linalg.inv(row['data_cov']),delta)
                consistent.append(np.allclose(radii,4,rtol=1e-9,atol=1e-9))
                plt.close(fig)
        led.check_ints('all four Figure 5 rendered ellipses match the stated boundary',consistent,[True]*4)
        led.check_ints('Figure 5 physical axis patterns agree with Supplementary Figure 8',
            [all(np.allclose(a,b) or np.allclose(a,-b) for a,b in zip(
                data['model']['nolens']['basis'].T,
                f.singular_pattern_table()['withlens']['patterns'][[2,4]])),
             np.allclose(data['measured']['nolens']['basis'],
                f.singular_pattern_table(measured=True)['nolens']['patterns'][[4,6]].T)],
            [True,True])

    def neural_controls():
        from figure_revision import prior_neural_results, prior_neural_cases
        saved=prior_neural_results();cases=prior_neural_cases()
        one('current neural hidden width',32,r'hidden layer of \$NUM\$ ReLU units')
        seq('current neural training steps and batch',[saved['steps'],saved['batch']],
            r'for \$NUM\$ steps with batches of \$NUM\$')
        seq('current neural seeds',saved['seeds'],r'three fixed seeds \(\$NUM\$, \$NUM\$, \$NUM\$\)')
        one('current neural independent validation size',saved['validation_count'],r'evaluated on \$NUM\$ independent draws')
        one('current neural displayed seed',saved['figure_seed'],r'seed \$NUM\$ is displayed without selection')
        records=[]
        for arm,row in saved['arms'].items():
            for key,items in row['validation'].items():
                records.extend(items)
                led.check_ints(arm+' '+key+' all declared training seeds',[v['seed'] for v in items],saved['seeds'])
                led.check_ints(arm+' '+key+' analytic coefficients match current model',
                    [np.allclose(cases[arm]['analytic'][key],row['analytic'][key],rtol=1e-12,atol=1e-12)],[True])
                shown=next(v for v in items if v['seed']==saved['figure_seed'])
                led.check_ints(arm+' '+key+' displayed output is fixed seed',row['neural'][key],shown['estimate'])
        led.check_ints('all current neural independent validation RMSE below 0.02',
            [len(records),all(max(v['heldout_rmse_vs_analytic'])<.02 for v in records)],[12,True])
        reproduced=[];approximates=[]
        for arm,row in saved['arms'].items():
            for key,items in row['validation'].items():
                for item in items:
                    metrics=checkpoint_metrics(OUT.parents[3]/'notebooks/models/nn_prior_current'/
                        f'{arm}_{key}_{item["seed"]}.pt',cases[arm]['priors'][key],
                        cases[arm]['noise_covariance'],cases[arm]['observation'],saved['validation_count'])
                    reproduced.append(metrics['geometry_sha256']==saved['geometry_sha256'] and all(
                        np.allclose(metrics[k],item[k],rtol=2e-5,atol=2e-8)
                        for k in ('estimate','heldout_rmse_vs_analytic','heldout_coefficient_mse')))
                    approximates.append(max(metrics['heldout_rmse_vs_analytic'])<.02)
        led.check_ints('current checkpoint weights reproduce all reported validation statistics',reproduced,[True]*12)
        led.check_ints('current loaded networks independently meet analytic-mean tolerance',approximates,[True]*12)
        with (OUT/'source_data/fig05_neural_comparison.csv').open(encoding='utf-8') as stream:
            rows=list(csv.DictReader(stream))
        for row in rows:
            arm=row['arm'];key=row['prior'];coeff=np.asarray(saved['arms'][arm]['neural'][key])
            # The existing Source Data writer exports six significant digits.
            error=np.linalg.norm(coeff-cases[arm]['truth'])/np.linalg.norm(cases[arm]['truth'])
            led.check_ints('neural full-profile error '+arm+' '+key,
                [float(f'{error:.6g}')],[float(row['relative_profile_error'])])

    for label,check in [('figure set and acquisition',figures_and_protocol),('model pair',pair),
        ('model tasks',model),('measured budget and cutoff conventions',measured_budget),
        ('criterion sensitivity',threshold_sensitivity),
        ('parameter choices and analysis settings',parameter_choices),
        ('recorded scalar/profile results',recorded),('fixed-physical-prior examples',priors),
        ('matched current neural controls',neural_controls)]:
        led.guarded(label,check)
