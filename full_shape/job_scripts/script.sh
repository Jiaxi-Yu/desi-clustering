for tracer in LRG3 ELG2 QSO1; do
    srun -n 4 python validation_systematic_templates.py \
        --todo sample \
        --nchains 4 \
        --stats mesh2_spectrum \
        --tracers "$tracer" \
        --sampler emcee \
        --dataset data-dr2-v2 \
        --fits_dir /pscratch/sd/a/adematti/fits_systematic_templates \
        --syst_templates auw amr ric
done