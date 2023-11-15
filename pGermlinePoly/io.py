from cyvcf2 import VCF 
from cerebrus import Validator
import yaml


def check_samples(vcf, samples=[]):
	"""Check that all of the requested samples are in the VCF."""
	for s in samples:
		assert s in vcf.samples


def check_anno(vcf, annotations=[]):
	"""Check that the annotations appropriately work."""
	for a in annotations:
		assert a in vcf.FORMAT

def create_germline_anno(vcf, germline_sample):
	"""Create a germline annotation for the clone sequencing data."""
	pass

germline_schema= {
  'ind': {
    'required': True,
    'type': 'string'
  },
  'sex': {
    'required': True,
    'type': 'string'
  },
  'age':{
  	'required': True,
  	'type': 'int'
  },
  'germline':{
  	'required': True
  	'type': 'list',
  	'schema': {
  		'value': {
  			'type': 'string'
  		}
  	}	
  },
  'clones': {
  	'required': True,
  	'type': 'list',
  	'schema':{
  		'value': {
  			'type': 'string'
  		}
  	}
  }
}

def validate_config(config_yaml_fp, schema=germline_schema):
	"""Validate a config-file. """
	v = Validator(schema)
	with open(config_yaml_fp, 'r') as stream:
    config = yaml.load(stream)
		assert v.validate(config)
	return config 