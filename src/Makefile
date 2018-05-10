all:
	$(NOOP)

clean:
	rm *.json

hcl:
	python3 hclustering.py ../dataset/AccidentsSet01.csv

hcl8:
	python3 hclustering.py ../dataset/AccidentsSet01.csv --threshold 8

avg8:
	python3 hclustering.py --link-method AVERAGE ../dataset/AccidentsSet01.csv --threshold 8

