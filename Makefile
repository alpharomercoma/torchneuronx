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

# The three targets above cover only results/{trn1,trn2,inf2}/. On-box drivers
# also pushed to per-lane and per-hostname prefixes -- results/trn1-specdec/,
# results/final-ip-172-31-20-190-specdec/, results/trn1-ppl/ and thirteen more.
# 736 objects lived ONLY in those, including the k=8 and k=16 arms of the
# speculative-decoding ladder and the int8 perplexity receipts. Mirror the whole
# prefix and reconcile by hand; never assume the three canonical ones are complete.
pull-results-all: ## mirror EVERY results/ prefix, including the per-lane ones
	aws s3 sync s3://$(BUCKET)/results/ .s3-mirror/results/

report:         ## regenerate comparison.json + REPORT tables from results/
	python3 analysis/make_report.py

test:           ## local gate 1: harness + CDK assertions (no AWS, no Neuron hardware)
	uvx --with pytest --with numpy pytest -q tests
	cd cdk && uv run pytest -q

synth:          ## synth all 4 stacks; npx avoids needing a global cdk install
	cd cdk && npx -y aws-cdk@2 synth

.PHONY: push-code pull-results pull-results-all report test synth
