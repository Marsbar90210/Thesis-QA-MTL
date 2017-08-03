ifeq ($(OS),Windows_NT)
    PYTHON = python
else
    PYTHON = sudo python3
endif

COMMAND = $(PYTHON) -W ignore
MODELS = stl stlpte mtl mtlpte stletetf stletetfct stletetfws stletetfpte stletetfptews mtletetf mtletetfiter mtletetfws mtletetfpte mtletetfptews
OUTMODELS = $(foreach m,$(MODELS),out$(m))
UPMODELS = $(foreach m,$(MODELS),upd$(m))
OUTPUTS = ./../outputs/output

list:
	@echo $(MODELS)

$(UPMODELS):
	$(PYTHON) updatetablescript.py $(subst upd,,$@) $(filter-out $@,$(MAKECMDGOALS))

$(MODELS):
	$(COMMAND) model$@.py $(filter-out $@,$(MAKECMDGOALS))

$(OUTMODELS):
	echo Output for model $(subst out,,$@) >> $(OUTPUTS)$(subst out,,$@).txt
	$(COMMAND) model$(subst out,,$@).py $(filter-out $@,$(MAKECMDGOALS)) >> $(OUTPUTS)$(subst out,,$@).txt

%:
	@:
