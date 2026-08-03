# Laptop-side conveniences. On-box work happens over SSM sessions.
BUCKET=neuron-pipelines-artifacts-600627330911

push-code:      ## sync harness to S3 (boxes pull with shared/bin/pull_code.sh)
	aws s3 sync . s3://$(BUCKET)/code/ --delete \
	  --exclude ".git/*" --exclude "*/results/*" --exclude "cdk/cdk.out/*" \
	  --exclude "cdk/.venv/*" --exclude ".venv/*" --exclude "*__pycache__*"

pull-results:   ## fetch raw triplets from all three boxes
	aws s3 sync s3://$(BUCKET)/results/trn1/ trn1/results/
	aws s3 sync s3://$(BUCKET)/results/trn2/ trn2/results/
	aws s3 sync s3://$(BUCKET)/results/inf2/ inf2/results/

report:         ## regenerate comparison.json + REPORT tables from results/
	python3 analysis/make_report.py

test:           ## local gate 1: harness + CDK assertions
	cd tests && python3 -m pytest -q ../tests
	cd cdk && uv run pytest -q

synth:
	cd cdk && uv run cdk synth

.PHONY: push-code pull-results report test synth
