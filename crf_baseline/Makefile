.DEFAULT_GOAL := all

VIRTUALENV := build/virtualenv

$(VIRTUALENV)/.installed: requirements.txt
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python python3 $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r requirements.txt
	touch $@

.PHONY: virtualenv
virtualenv: $(VIRTUALENV)/.installed

all: virtualenv
