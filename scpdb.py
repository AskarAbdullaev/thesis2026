import os
import json
import warnings
from collections import Counter

import pandas as pd
import numpy as np
import regex as re


from initial_tools import (map_pdb_to_ec_number,
                           map_pdb_to_scop,
                           map_pdb_to_uniprot,
                           map_uniprot_to_uniref)


pd.options.display.max_colwidth = 1000
pd.options.display.max_columns = 20
warnings.simplefilter("ignore")

###############################################
# SCPDB Class and associated instances
###############################################

class SCPDB():
    """
    Overall class that points to the folder
    """

    RESIDUES = ['GLY', 'LEU', 'ILE', 'VAL', 'ALA', 'PRO', 'ASP', 'ASN', 'GLU', 'GLN',
                'ARG', 'LYS', 'HIS', 'CYS', 'MET', 'SER', 'THR', 'TYR', 'TRP', 'PHE']
    
    ONE_LETTER = {'GLY': 'G', 'LEU': 'L', 'ILE': 'I', 'VAL': 'V', 'ALA': 'A', 'PRO': 'P',
                  'ASP': 'D', 'ASN': 'N', 'GLU': 'E', 'GLN': 'Q',
                  'ARG': 'R', 'LYS': 'K', 'HIS': 'H',
                  'CYS': 'C', 'MET': 'M', 'SER': 'S', 'THR': 'T',
                  'TYR': 'Y', 'TRP': 'W', 'PHE': 'F'}
    
    MW = {"H": 1, "C": 12, "N": 14, "O": 16, "S": 32, "P": 31, "Mg": 24.3, "Zn": 65.4, "Cl": 35.5, "Br": 79.9, "I": 126.9,
          "Fe": 55.8, "Ca": 40.1, "Mn": 54.9, "Cu": 63.5, "Co": 58.9, "Ni": 58.7, "Na": 23, "K": 39.1}
    
    RESIDUES_GROUPS = {'GLY': 'ALIPHATIC', 'LEU': 'ALIPHATIC', 'ILE': 'ALIPHATIC', 'VAL': 'ALIPHATIC',
                       'ALA': 'ALIPHATIC', 'PRO': 'ALIPHATIC', 'ASP': 'NEGATIVELY CHARGED', 
                       'ASN': 'POLAR UNCHARGED', 'GLU': 'NEGATIVELY CHARGED', 'GLN': 'POLAR UNCHARGED',
                       'ARG': 'POSITIVELY CHARGED', 'LYS': 'POSITIVELY CHARGED', 'HIS': 'POSITIVELY CHARGED',
                        'CYS': 'POLAR UNCHARGED', 'MET': 'ALIPHATIC', 'SER': 'POLAR UNCHARGED', 'THR': 'POLAR UNCHARGED',
                        'TYR': 'AROMATIC', 'TRP': 'AROMATIC', 'PHE': 'AROMATIC'}

    def __init__(self,
                 scpdb_id: str,
                 dir: str = 'Files/scPDB'):

        # Check the input
        assert isinstance(dir, str), f'dir must be str, not {type(dir)}'
        assert isinstance(scpdb_id, str), f'scpdb_id must be str, not {type(scpdb_id)}'
        assert os.path.isdir(dir), f'no such directory: {dir}'

        self.dir = dir

        folders_in_dir = os.listdir(dir)
        if '_' in scpdb_id:
            assert scpdb_id in folders_in_dir, f'scpdb entry: {scpdb_id} is not found in the directory: {dir}'
            self.pdb, self.version = scpdb_id.split('_')
            self.scpdb = scpdb_id
        else:
            candidates = list(filter(lambda x: x.split('_')[0] == scpdb_id, folders_in_dir))
            if len(candidates) == 0:
                raise AssertionError(f'pdb id {scpdb_id} is not found in the directory: {dir}')
            elif len(candidates) > 1:
                raise AssertionError(f'pdb id {scpdb_id} matches several folders in the directory: {candidates}')
            else:
                self.pdb, self.version = candidates[0].split('_')
                self.scpdb = candidates[0]

        self.folder_path = os.path.join(dir, self.scpdb)
        self.available_files = os.listdir(self.folder_path)

        self._folder = BasicInfo(self)
        self._html = HTML(self)
        self._mol2 = Mol2(self)
        self._rcsb = PDB(self)
        self._uniprot = UniProt(self, self.folder.uniprot_json)
        self._uniprots = [UniProt(self, up_path) for up_path in self.folder.all_uniprot_jsons]

        up_html = self.html.protein.Uniprot_ID
        up_basic = self.folder.UniProt
        up_exact = self.uniprot.AC
        up_rcsb = self.rcsb.Dominant_Chain.UniProt_ID if self.rcsb.Dominant_Chain else None

        self._up_combined = None
        if up_exact:
            self._up_combined = up_exact
        elif up_basic:
            self._up_combined = up_basic
        elif up_html:
            self._up_combined = up_html
        else:
            self._up_combined = up_rcsb

        ec_html = self.html.protein.EC
        ec_basic = self.folder.EC_Number
        ec_exact = self.uniprot.EC
        
        self._ec_combined = None
        if ec_exact:
            self._ec_combined = ec_exact
        elif ec_basic:
            self._ec_combined = ec_basic
        else:
            self._ec_combined = ec_html

        self._scope = self.html.protein.SCOPe_Dominant_Chain
        self._scop1 = self.folder.SCOP1
        self._scop2 = self.folder.SCOP2
        pass

    @property
    def dict(self):

        _dict = {'UniProt': self.UniProt, 'EC': self.EC, 'SCOPe': self.SCOPe, 'SCOP1': self.SCOP1, 'SCOP2': self.SCOP2}
        _dict.update(self.folder.dict_with_prefix)
        _dict.update(self.uniprot.dict_with_prefix)
        _dict.update(self.html.dict_with_prefix)
        _dict.update(self.rcsb.dict_complete)
        _dict.update(self.mol2.dict_with_prefix)

        return _dict

    def __str__(self):

        return f'<Entry {self.scpdb}>'
    
    def __repr__(self):

        return str(self)
    
    @property
    def folder(self):
        return self._folder
    
    @property
    def uniprot(self):
        return self._uniprot
    
    @property
    def all_uniprots(self):
        return self._uniprots
    
    @property
    def html(self):
        return self._html
    
    @property
    def rcsb(self):
        return self._rcsb
    
    @property
    def mol2(self):
        return self._mol2
    
    @property
    def UniProt(self):
        return self._up_combined
    
    @property
    def EC(self):
        return self._ec_combined
    
    @property
    def SCOPe(self):
        return self._scope
    
    @property
    def SCOP1(self):
        return self._scop1
    
    @property
    def SCOP2(self):
        return self._scop2


class BasicInfo():
    """
    Collection of properties
    """

    def __init__(self, parent: SCPDB):

        self._scpdb = parent

        # Check basic files

        if 'protein.mol2' not in self._scpdb.available_files:
            self._protein_mol2_path = None
        else:
            self._protein_mol2_path = os.path.join(self._scpdb.folder_path, 'protein.mol2')
            with open(self._protein_mol2_path, 'r') as f:
                f = f.read()
            if 'html' in f:
                self._protein_mol2_path = None

        if 'ligand.mol2' not in self._scpdb.available_files:
            self._ligand_mol2_path = None
        else:
            self._ligand_mol2_path = os.path.join(self._scpdb.folder_path, 'ligand.mol2')
            with open(self._ligand_mol2_path, 'r') as f:
                f = f.read()
            if 'html' in f:
                self._ligand_mol2_path = None

        if 'cavity.mol2' not in self._scpdb.available_files:
            self._cavity_mol2_path = None
        else:
            self._cavity_mol2_path = os.path.join(self._scpdb.folder_path, 'cavity.mol2')
            with open(self._cavity_mol2_path, 'r') as f:
                f = f.read()
            if 'html' in f:
                self._cavity_mol2_path = None

        if 'IPF.txt' not in self._scpdb.available_files:
            self._ipf_txt_path = None
        else:
            self._ipf_txt_path = os.path.join(self._scpdb.folder_path, 'IPF.txt')
            with open(self._ipf_txt_path, 'r') as f:
                f = f.read()
            if 'html' in f:
                self._ipf_txt_path = None

        if 'interaction.mol2' not in self._scpdb.available_files:
            self._interaction_mol2_path = None
        else:
            self._interaction_mol2_path = os.path.join(self._scpdb.folder_path, 'interaction.mol2')
            with open(self._interaction_mol2_path, 'r') as f:
                f = f.read()
            if 'html' in f:
                self._interaction_mol2_path = None

        if 'site.mol2' not in self._scpdb.available_files:
            self._site_mol2_path = None
        else:
            self._site_mol2_path = os.path.join(self._scpdb.folder_path, 'site.mol2')
            with open(self._site_mol2_path, 'r') as f:
                f = f.read()
            if 'html' in f:
                self._site_mol2_path = None

        # Check web files

        if 'html.txt' not in self._scpdb.available_files:
            self._html_txt_path = None
        else:
            self._html_txt_path = os.path.join(self._scpdb.folder_path, 'html.txt')

        if 'html.json' not in self._scpdb.available_files:
            self._html_json_path = None
        else:
            self._html_json_path = os.path.join(self._scpdb.folder_path, 'html.json')

        # Check RCSB fetches

        if 'rcsb_entry.json' not in self._scpdb.available_files:
            self._rcsb_entry_json_path = None
        else:
            self._rcsb_entry_json_path = os.path.join(self._scpdb.folder_path, 'rcsb_entry.json')

        if 'rcsb_identifiers.json' not in self._scpdb.available_files:
            self._rcsb_identifiers_json_path = None
        else:
            self._rcsb_identifiers_json_path = os.path.join(self._scpdb.folder_path, 'rcsb_identifiers.json')

        self._rcsb_entity_json_paths = list(filter(lambda x: 'rcsb' in x and 'entity' in x and not 'chain' in x, self._scpdb.available_files))
        self._rcsb_chain_json_paths = list(filter(lambda x: 'rcsb' in x and 'chain' in x, self._scpdb.available_files))
        self._rcsb_assembly_json_paths = list(filter(lambda x: 'rcsb' in x and 'assembly' in x, self._scpdb.available_files))

        if self._rcsb_identifiers_json_path and self._rcsb_entry_json_path:

            with open(self._rcsb_identifiers_json_path, 'rb') as f:
                self._rcsb_identifiers = json.load(f)
        else:
            self._rcsb_identifiers = {}

        self._rcsb_entities_theory = self._rcsb_identifiers.get('entity_ids', [])
        self._rcsb_assemblies_theory = self._rcsb_identifiers.get('assembly_ids', [])
        self._rcsb_non_polymer_entities_theory = self._rcsb_identifiers.get('non_polymer_entity_ids', [])
        self._rcsb_polymer_entities_theory = self._rcsb_identifiers.get('polymer_entity_ids', [])
        self._rcsb_branched_entities_theory = self._rcsb_identifiers.get('branched_entity_ids', [])

        self._rcsb_entities_loaded = list(map(lambda x: x.replace('.json', '').replace('rcsb_entity_', ''), self._rcsb_entity_json_paths))
        self._rcsb_assemblies_loaded = list(map(lambda x: x.replace('.json', '').replace('rcsb_assembly_', ''), self._rcsb_assembly_json_paths))
        self._rcsb_non_polymer_entities_loaded = list(set(self._rcsb_non_polymer_entities_theory) & set(self._rcsb_entities_loaded))
        self._rcsb_polymer_entities_loaded = list(set(self._rcsb_polymer_entities_theory) & set(self._rcsb_entities_loaded))
        self._rcsb_branched_entities_loaded = list(set(self._rcsb_branched_entities_theory) & set(self._rcsb_entities_loaded))

        if not self._rcsb_identifiers:
            self._rcsb_identifiers = None

        self._ups = map_pdb_to_uniprot(pdb_id=self._scpdb.pdb, per_chain=False, mode='all')
        self._up = map_pdb_to_uniprot(pdb_id=self._scpdb.pdb, per_chain=False, mode='longest')
        self._ec = map_pdb_to_ec_number(pdb_id=self._scpdb.pdb, per_chain=False, mode='first')
        self._ec_split = [] if not self._ec else self._ec.split('.')
        self._scop1 = map_pdb_to_scop(pdb_id=self._scpdb.pdb, per_chain=False, mode='longest', level=0, as_names=True)
        self._scop2 = map_pdb_to_scop(pdb_id=self._scpdb.pdb, per_chain=False, mode='longest', level=1, as_names=True)
        self._scop3 = map_pdb_to_scop(pdb_id=self._scpdb.pdb, per_chain=False, mode='longest', level=2, as_names=True)
        self._scop4 = map_pdb_to_scop(pdb_id=self._scpdb.pdb, per_chain=False, mode='longest', level=3, as_names=True)
        self._scop5 = map_pdb_to_scop(pdb_id=self._scpdb.pdb, per_chain=False, mode='longest', level=4, as_names=True)

        if self._up:
            self._uniref50 = map_uniprot_to_uniref(uniprot=self._up, mode='largest', level=50, add_cluster_names=True)
            if self._uniref50:
                self._uniref50, self._uniref50_name = self._uniref50
            self._uniref90 = map_uniprot_to_uniref(uniprot=self._up, mode='largest', level=90, add_cluster_names=True)
            if self._uniref90:
                self._uniref90, self._uniref90_name = self._uniref90
            self._uniref100 = map_uniprot_to_uniref(uniprot=self._up, mode='largest', level=100, add_cluster_names=True)
            if self._uniref100:
                self._uniref100, self._uniref100_name = self._uniref100

            if f'{self._up.upper()}.json' not in self._scpdb.available_files:
                self._uniprot_json_path = None
            else:
                self._uniprot_json_path = os.path.join(self._scpdb.folder_path, f'{self._up.upper()}.json')
        else:
            self._uniref50, self._uniref50_name = None, None
            self._uniref90, self._uniref90_name = None, None
            self._uniref100, self._uniref100_name = None, None
            self._uniprot_json_path = None

        if not self._ups:
            self._all_uniprot_jsons_paths = []
        else:
            self._all_uniprot_jsons_paths = [os.path.join(self._scpdb.folder_path, f'{up.upper()}.json') for up in self._ups
                                             if f'{up}.json' in self._scpdb.available_files]

        self._dict = {'scPDB ID': self._scpdb.scpdb, 'PDB ID': self._scpdb.pdb, 'Protein Mol2': self.protein_mol2.exists,
                      'Site Mol2': self.site_mol2.exists, 'Cavity Mol2': self.cavity_mol2.exists,
                      'Interaction Mol2': self.interaction_mol2.exists, 'IPF txt': self.ipf_txt.exists,
                      'HTML JSON': self.html_json.exists, 'HTML txt': self.html_txt.exists,
                      'RCSB JSON': self.rcsb_entry.exists, 'RCSB identifiers': self.rcsb_identifiers.exists,
                      'RCSB Entities': len(self.rcsb_entities), 'RCSB Assemblies': len(self.rcsb_assemblies),
                      'RCSB Poly Entities': len(self.rcsb_polymers), 'RCSB Non Poly': len(self.rcsb_nonpolymers),
                      'RCSB Branched': len(self.rcsb_branched), 'Minimum Requirements': self.minimum_requirements,
                      'EC': self.EC_Number, 'EC1': self.EC1, 'EC2': self.EC2, 'EC3': self.EC3, 'EC4': self.EC4,
                      'EC12': self.EC12,  'EC123': self.EC123,
                      'UniProt': self.UniProt, 'All UniProts': self._ups, 'SCOP1': self.SCOP1, 'SCOP2': self.SCOP2,
                      'SCOP3': self.SCOP3, 'SCOP4': self.SCOP4, 'SCOP5': self.SCOP5, 'UniRef50': self.UniRef50,
                      'UniRef50_Name': self.UniRef50_Name, 'UniRef90': self.UniRef90, 'UniRef90_Name': self.UniRef90_Name,
                      'UniRef100': self.UniRef100, 'UniRef100_Name': self.UniRef100_Name}
        
        self._prefix = 'basic$'
        

    def __str__(self):

        return self._scpdb.folder_path
    
    def __repr__(self):

        return str(self)
    
    @property
    def dict(self):
        return self._dict
    
    @property
    def dict_with_prefix(self):
        return {self._prefix + k: self._dict[k] for k in self._dict}

    @property
    def EC_Number(self):
        return self._ec
    
    @property
    def EC1(self):
        return self._ec_split[0] if len(self._ec_split) > 0 else None
    
    @property
    def EC2(self):
        return self._ec_split[1] if len(self._ec_split) > 1 else None
    
    @property
    def EC12(self):
        return '.'.join(self._ec_split[:2]) if len(self._ec_split) > 1 else None
    
    @property
    def EC3(self):
        return self._ec_split[2] if len(self._ec_split) > 2 else None
    
    @property
    def EC123(self):
        return '.'.join(self._ec_split[:3]) if len(self._ec_split) > 2 else None
    
    @property
    def EC4(self):
        return self._ec_split[3] if len(self._ec_split) > 3 else None
    
    @property
    def UniProt(self):
        return self._up
    
    @property
    def All_UniProts(self):
        return self._ups
    
    @property
    def SCOP1(self):
        return self._scop1
    
    @property
    def SCOP2(self):
        return self._scop2
    
    @property
    def SCOP3(self):
        return self._scop3
    
    @property
    def SCOP4(self):
        return self._scop4
    
    @property
    def SCOP5(self):
        return self._scop5
    
    @property
    def UniRef50(self):
        return self._uniref50
    
    @property
    def UniRef90(self):
        return self._uniref90

    @property
    def UniRef100(self):
        return self._uniref100
    

    @property
    def UniRef50_Name(self):
        return self._uniref50_name
    
    @property
    def UniRef90_Name(self):
        return self._uniref90_name

    @property
    def UniRef100_Name(self):
        return self._uniref100_name
    
    
    @property
    def minimum_requirements(self):
        return self.protein_mol2.exists & self.site_mol2.exists

    # Get the main files
    @property
    def protein_mol2(self):
        return Path(self._protein_mol2_path)
    
    @property
    def site_mol2(self):
        return Path(self._site_mol2_path)
    
    @property
    def cavity_mol2(self):
        return Path(self._cavity_mol2_path)
    
    @property
    def interaction_mol2(self):
        return Path(self._interaction_mol2_path)
    
    @property
    def ipf_txt(self):
        return Path(self._ipf_txt_path)
    
    @property
    def ligand_mol2(self):
        return Path(self._ligand_mol2_path)
    
    @property
    def html_txt(self):
        return Path(self._html_txt_path)
    
    @property
    def html_json(self):
        return Path(self._html_json_path)
    
    @property
    def uniprot_json(self):
        return Path(self._uniprot_json_path)
    
    @property
    def all_uniprot_jsons(self):
        return [Path(up) for up in self._all_uniprot_jsons_paths]
    
    @property
    def Uniprots_Found(self):
        return len(self.all_uniprot_jsons)
    
    @property
    def rcsb_entry(self):
        return Path(self._rcsb_entry_json_path)
    
    @property
    def rcsb_identifiers(self):
        return Path(self._rcsb_identifiers_json_path)
    
    # Get loaded RCSB sub files:

    @property
    def rcsb_assemblies(self):
        return [Path(os.path.join(self._scpdb.folder_path, f'rcsb_assembly_{p}.json'))
                for p in self._rcsb_assemblies_loaded]
    
    @property
    def rcsb_entities(self):
        return [Path(os.path.join(self._scpdb.folder_path, f'rcsb_entity_{p}.json'))
                for p in self._rcsb_entities_loaded]
    
    @property
    def rcsb_polymers(self):
        return [Path(os.path.join(self._scpdb.folder_path, f'rcsb_entity_{p}.json'))
                 for p in self._rcsb_polymer_entities_loaded]
    
    @property
    def rcsb_nonpolymers(self):
        return [Path(os.path.join(self._scpdb.folder_path, f'rcsb_entity_{p}.json'))
                for p in self._rcsb_non_polymer_entities_loaded]
    
    @property
    def rcsb_branched(self):
        return [Path(os.path.join(self._scpdb.folder_path, f'rcsb_entity_{p}.json'))
                for p in self._rcsb_branched_entities_loaded]
    
    # Get theoretical lists of sub identifiers
    @property
    def rcsb_assemblies_theory(self):
        return self._rcsb_assemblies_theory
    
    @property
    def rcsb_entities_theory(self):
        return self._rcsb_entities_theory
    
    @property
    def rcsb_polymers_theory(self):
        return self._rcsb_polymer_entities_theory
    
    @property
    def rcsb_nonpolymers_theory(self):
        return self._rcsb_non_polymer_entities_theory
    
    @property
    def rcsb_branched_theory(self):
        return self._rcsb_branched_entities_theory
    
    # Get exact sub file
    def entity(self, number: str | int):

        if not str(number) in self._rcsb_entities_loaded:
            return None
        return Path(os.path.join(self._scpdb.folder_path, f'rcsb_entity_{number}.json'))
    
    def assembly(self, number: str | int):

        if not str(number) in self._rcsb_assemblies_loaded:
            return None
        return Path(os.path.join(self._scpdb.folder_path, f'rcsb_assembly_{number}.json'))


class UniProt():
    """
    Collection of properties
    """

    def __init__(self, parent: SCPDB, file = None):

        self._scpdb = parent

        # Check basic files

        if not file.content:
            self._dict = {}
            self._ac = None
            self._ec = None
            return

        d = file.content

        self._chain = d['Chain']
        self._is_main = d['IsMain']
        self._ac = d.get('primaryAccession', None)
        self._id = d.get('uniProtkbId', None)
        self._annotation_score = d.get('annotationScore', None)

        organism = d.get('organism', None)
        if organism is not None:
            self._organism = organism.get('scientificName', None)
            self._organism_common_name = organism.get('commonName', None)
            self._organism_id = organism.get('taxonId', None)
            self._organism_lineage = organism.get('lineage', None)
        else:
            self._organism, self._organism_common_name, self._organism_id, self._organism_lineage = None, None, None, None

        self._protein_existance = d.get('proteinExistence', None)

        description = d.get('proteinDescription', None)
        self._ec, self._protein_name, self._alt_names = None, None, None
        if description is not None:

            if 'recommendedName' in description:
                self._protein_name = description['recommendedName']['fullName']['value']
            else:
                self._protein_name = description["submissionNames"][0]["fullName"]['value']
            ec = description.get('ecNumbers', None)
            if ec:
                self._ec = ec[0]['value']
            else:
                self._ec = None
            alt_names = description.get('alternativeNames', None)
            if alt_names:
                self._alt_names = [alt['fullName']['value'] for alt in alt_names]
            else:
                self._alt_names = None
        
        gene = d.get('genes', None)
        self._gene = None
        if gene:
            self._gene = gene[0]

            if 'geneName' in gene[0]:
                self._gene = gene[0]['geneName']['value']
            elif "orderedLocusNames" in gene[0]:
                self._gene = gene[0]["orderedLocusNames"][0]['value']
            else:
                self._gene = None
        else:
            self._gene = None

        comments = d.get('comments', None)
        self._ec_numbers = []
        self._functions = []
        self._reactions = []
        self._subunits = []
        self._similarities = []
        self._locations = []
        if comments:
            for com in comments:
                comment_type = com['commentType']

                if comment_type == 'FUNCTION':
                    self._functions.append(com['texts'][0]['value'])

                elif comment_type == 'CATALYTIC ACTIVITY':
                    self._ec_numbers.append(com.get('reaction', {}).get('ecNumber', None))
                    self._reactions.append(com.get('reaction', {}).get('name', None))
                
                elif comment_type == 'SUBUNIT':
                    self._subunits.append(com['texts'][0]['value'])

                elif comment_type == 'SIMILARITY':
                    self._similarities.append(com['texts'][0]['value'])

                elif comment_type == 'SUBCELLULAR LOCATION':

                    if 'subcellularLocations' in com:
                        self._locations.extend([loc['location']['value'] for loc in com['subcellularLocations']])

        self._ec_numbers = sorted(set(filter(lambda x: x, self._ec_numbers)))
        self._functions = sorted(set(filter(lambda x: x, self._functions)))
        self._reactions = sorted(set(filter(lambda x: x, self._reactions)))
        self._subunits = sorted(set(filter(lambda x: x, self._subunits)))
        self._similarities = sorted(set(filter(lambda x: x, self._similarities)))
        self._locations = sorted(set(filter(lambda x: x, self._locations)))

        if not self._ec and self._ec_numbers:
            self._ec = self._ec_numbers[0]


        features = d.get('features', None)
        self._chains = []
        self._domains = []
        self._active_sites = []
        self._binding_sites = []
        self._beta_strands = []
        self._helices = []
        self._turns = []
        if features:
            for feat in features:

                feature_type = feat['type']

                if feature_type == 'Chain':
                    end = feat['location']['end']['value']
                    start = feat['location']['start']['value']
                    if isinstance(end, int) and isinstance(start, int):
                        length = end - start + 1
                    else:
                        length = None
                    self._chains.append(
                        {'length': length, 'description': feat['description']}
                    )

                elif feature_type == 'Domain':
                    end = feat['location']['end']['value']
                    start = feat['location']['start']['value']
                    if isinstance(end, int) and isinstance(start, int):
                        length = end - start + 1
                    else:
                        length = None
                    self._domains.append(
                        {'length': length, 'description': feat['description']}
                    )

                elif feature_type == 'Active site':
                    end = feat['location']['end']['value']
                    start = feat['location']['start']['value']
                    if isinstance(end, int) and isinstance(start, int):
                        length = end - start + 1
                    else:
                        length = None
                    self._active_sites.append(
                        {'length': length, 'description': feat['description']}
                    )

                elif feature_type == 'Binding site':
                    end = feat['location']['end']['value']
                    start = feat['location']['start']['value']
                    if isinstance(end, int) and isinstance(start, int):
                        length = end - start + 1
                    else:
                        length = None
                    self._binding_sites.append({
                        'start': feat['location']['start']['value'],
                        'end': feat['location']['end']['value'],
                        'length': feat['location']['end']['value'] - feat['location']['start']['value'],
                        'description': feat['description'],
                        'ligand': feat['ligand']['name']
                    })
                
                elif feature_type == 'Beta strand':
                    self._beta_strands.append({
                        'start': feat['location']['start']['value'],
                        'end': feat['location']['end']['value']
                    })

                elif feature_type == 'Helix':
                    self._helices.append({
                        'start': feat['location']['start']['value'],
                        'end': feat['location']['end']['value']
                    })

                elif feature_type == 'Turn':
                    self._turns.append({
                        'start': feat['location']['start']['value'],
                        'end': feat['location']['end']['value']
                    })

        sequence = d.get('sequence', None)
        if sequence:
            self._fasta = sequence['value']
            self._length = sequence['length']
            self._mw = sequence['molWeight']
                
        self._ = d.get('uniProtkbId', None)
        self._id = d.get('uniProtkbId', None)
        self._id = d.get('uniProtkbId', None)
        self._id = d.get('uniProtkbId', None)


        self._dict = {'scPDB ID': self._scpdb.scpdb, 'PDB ID': self._scpdb.pdb, 'Chain Name': self.Chain, 'Is Main': self.IsMain,
                      'AC': self.AC, 'UniProt ID': self.ID, 'Anntation Score': self.Annotation_Score, 'Alternative Names': self.Alternative_Names,
                      'FASTA': self.FASTA, 'Length': self.Length, 'EC': self.EC, 'Gene': self.Gene, 'Existance': self.Existance,
                      'Subcellular Locations': self.Subcellular_Locations, 'MW': self.MW, 'Name': self.Protein_Name, 'Organism': self.Organism,
                      'Organism Common Name': self.Organism_Common_Name, 'Taxon ID': self.Organism_ID}
        
        self._prefix = 'uniprot$'

        

    def __str__(self):

        return f'<{self.Chain}: {self.AC}>'
    
    def __repr__(self):

        return str(self)
    
    @property
    def dict(self):
        return self._dict
    
    @property
    def dict_with_prefix(self):
        return {self._prefix + k: self._dict[k] for k in self._dict}
    
    @property
    def Chain(self):
        return self._chain
    
    @property
    def IsMain(self):
        return self._is_main
    
    @property
    def AC(self):
        return self._ac
    
    @property
    def ID(self):
        return self._id
    
    @property
    def Annotation_Score(self):
        return self._annotation_score
    
    @property
    def Active_Sites(self):
        return self._active_sites
    
    @property
    def Alternative_Names(self):
        return self._alt_names
    
    @property
    def Beta_Strands(self):
        return self._beta_strands
    
    @property
    def Binding_Sites(self):
        return self._binding_sites
    
    @property
    def Chains(self):
        return self._chains
    
    @property
    def Domains(self):
        return self._domains
    
    @property
    def FASTA(self):
        return self._fasta
    
    @property
    def EC(self):
        return self._ec
    
    @property
    def EC_Numbers(self):
        return self._ec_numbers
    
    @property
    def Functions(self):
        return self._functions
    
    @property
    def Gene(self):
        return self._gene
    
    @property
    def Helices(self):
        return self._helices
    
    @property
    def Length(self):
        return self._length
    
    @property
    def Subcellular_Locations(self):
        return self._locations
    
    @property
    def MW(self):
        return self._mw
    
    @property
    def Organism(self):
        return self._organism
    
    @property
    def Organism_Common_Name(self):
        return self._organism_common_name

    @property
    def Organism_ID(self):
        return self._organism_id
    
    @property
    def Organism_Lineage(self):
        return self._organism_lineage
    
    @property
    def Existance(self):
        return self._protein_existance
    
    @property
    def Protein_Name(self):
        return self._protein_name
    
    @property
    def Reactions(self):
        return self._reactions
    
    @property
    def Similarities(self):
        return self._similarities
    
    @property
    def Subunits(self):
        return self._subunits
    
    @property
    def Turns(self):
        return self._turns


class Path():

    def __init__(self, path: str | None):
        self._path = path

    def __str__(self):

        return 'NaN' if self.path is None else self.path
    
    def __repr__(self):

        return str(self)

    @property
    def path(self):

        return self._path
    
    @property
    def extention(self):

        if self.path is None:
            return None
        
        return self.path.split('.')[-1]
    
    @property
    def tail(self):

        if self.path is None:
            return None
        
        return self.path.split('/')[-1]
    
    @property
    def exists(self):

        if self.path is None:
            return None

        return os.path.isfile(self.path)
    
    @property
    def content(self):

        if self.path is None:
            return None
        
        
        with open(self.path, 'r') as f:
            if self.extention != 'json':
                cont = f.read()
            else:
                cont = json.load(f)
        
        return cont


class HTML():

    def __init__(self, parent: SCPDB):

        self._scpdb = parent

        if not self._scpdb.folder.html_json:
            self._path = None
        else:
            self._path = self._scpdb.folder.html_json.path

        self._dict = self._scpdb.folder.html_json.content if self._scpdb.folder.html_json.content is not None else {}
        self._protein = HTML_Protein(self)
        self._ligand = HTML_Ligand(self)

        if not self._dict:
            self._dict = {}
            return

        percentage_per_chain = self._dict.get('Percentage of Residues within binding site per Chain', None)
        if isinstance(percentage_per_chain, str):
            self._dict.update({'Chain Site Distribution': "/".join(sorted([x.split(' = ')[1] for x in percentage_per_chain.split(' / ')], reverse=True))})
            self._dict.update({'Dominant Chain': sorted([x.split(' = ') for x in percentage_per_chain.split(' / ')], key=lambda x: x[1], reverse=True)[0][0]})

        ec = self._dict.get('EC', None)
        if isinstance(ec, str) and len(ec) > 0 and ec != '/':
            self._dict.update({'EC1': ec.split('.')[0],
                               'EC2': ec.split('.')[1] if len(ec.split('.')) > 1 else None,
                               'EC3': ec.split('.')[2] if len(ec.split('.')) > 2 else None,
                               'EC4': ec.split('.')[3] if len(ec.split('.')) > 3 else None,
                               'EC12': ".".join(ec.split('.')[:2]) if len(ec.split('.')) > 1 else None,
                               'EC123': ".".join(ec.split('.')[:3]) if len(ec.split('.')) > 2 else None
                               })
        
        scope = self._dict.get('SCOPe Chain Classes', None)
        if scope and isinstance(scope, str):
            scope = {s.split(' ')[0]:s.split(' ')[1] for s in scope.split(' / ')}
            dominant_chain = self._dict.get('Dominant Chain', None)
            scope_dominant = scope.get(dominant_chain, None)
            if isinstance(scope_dominant, str):
                self._dict.update({'SCOPe Dominant Chain': scope_dominant,
                                    'SCOPe1 Dominant Chain': scope_dominant.split('.')[0],
                                    'SCOPe2 Dominant Chain': scope_dominant.split('.')[1] if len(scope_dominant.split('.')) > 1 else None,
                                    'SCOPe3 Dominant Chain': scope_dominant.split('.')[2] if len(scope_dominant.split('.')) > 2 else None,
                                    'SCOPe4 Dominant Chain': scope_dominant.split('.')[3] if len(scope_dominant.split('.')) > 3 else None,
                                    'SCOPe12 Dominant Chain': ".".join(scope_dominant.split('.')[:2]) if len(scope_dominant.split('.')) > 1 else None,
                                    'SCOPe123 Dominant Chain': ".".join(scope_dominant.split('.')[:3]) if len(scope_dominant.split('.')) > 2 else None
                                    })
        self._prefix = 'html$'
                
        
    def __str__(self):
        return f'<HTML {self._scpdb.scpdb}>'
    
    def __repr__(self):
        return str(self)

    @property
    def exists(self):
        return self.path.exists
    
    @property
    def path(self):
        return self._scpdb.folder.html_json.path

    @property
    def dict(self):
        return self._dict
    
    @property
    def dict_with_prefix(self):
        return {self._prefix + k: self._dict[k] for k in self._dict}
    
    @property
    def protein(self):
        return self._protein
    
    @property
    def ligand(self):
        return self._ligand
    
    @property
    def scPDB_URL(self):
        return self.dict.get('scPDB URL', None)
    
    @property
    def scPDB_ID(self):
        return self._scpdb.scpdb
    
    @property
    def PDB_ID(self):
        return self._scpdb.pdb

    @property
    def Protein_Mol2_Download_URL(self):
        return self.dict.get('Protein Mol2 Download URL', None)

    @property
    def Ligand_Mol2_Download_URL(self):
        return self.dict.get('Ligand Mol2 Download URL', None)

    @property
    def Site_Mol2_Download_URL(self):
        return self.dict.get('Site Mol2 Download URL', None)

    @property
    def IFP_Download_URL(self):
        return self.dict.get('IFP Download URL', None)

    @property
    def Cavity_Download_URL(self):
        return self.dict.get('Cavity Download URL', None)

    @property
    def Interaction_Download_URL(self):
        return self.dict.get('Interaction Download URL', None)

    @property
    def All_files_Download_URL(self):
        return self.dict.get('All files Download URL', None)


class HTML_Ligand():

    def __init__(self, parent: HTML):

        self._parent = parent
        self._properties = {'scPDB ID', 'PDB ID', 'Ligand Het Code', 'Ligand Formula', 'Ligand Molecular Weight', 'Ligand DrugBank ID',
                            'Ligand Buried Surface Area (A^2)', 'Ligand Polar Surface Area (A^2)', 'Ligand H-Bond Acceptors',
                            'Ligand H-Bond Donors', 'Ligand Rings', 'Ligand Aromatic Rings', 'Ligand Anionic Atoms', 'Ligand Cationic Atoms',
                            'Ligand Rule of Five Violation', 'Ligand Rotatable Bonds', 'Ligand Mass Center (X)', 'Ligand Mass Center (Y)',
                            'Ligand Mass Center (Z)'}

    def __str__(self):
        return f'<HTML Ligand {self.Het_Code} of {self._parent._scpdb.scpdb}>'
    
    def __repr__(self):
        return str(self)
    
    @property
    def dict(self):
        return {k: self._parent.dict[k] for k in self._parent.dict if k in self._properties}

    @property
    def Het_Code(self):
        return self._parent.dict.get('Ligand Het Code', None)

    @property
    def Formula(self):
        return self._parent.dict.get('Ligand Formula', None)

    @property
    def Molecular_Weight(self):
        return self._parent.dict.get('Ligand Molecular Weight', None)

    @property
    def DrugBank_ID(self):
        return self._parent.dict.get('Ligand DrugBank ID', None)

    @property
    def Buried_Surface_Area(self):
        return self._parent.dict.get('Ligand Buried Surface Area (A^2)', None)

    @property
    def Polar_Surface_Area(self):
        return self._parent.dict.get('Ligand Polar Surface Area (A^2)', None)

    @property
    def H_Bond_Acceptors(self):
        return self._parent.dict.get('Ligand H-Bond Acceptors', None)

    @property
    def H_Bond_Donors(self):
        return self._parent.dict.get('Ligand H-Bond Donors', None)

    @property
    def Rings(self):
        return self._parent.dict.get('Ligand Rings', None)

    @property
    def Aromatic_Rings(self):
        return self._parent.dict.get('Ligand Aromatic Rings', None)

    @property
    def Anionic_Atoms(self):
        return self._parent.dict.get('Ligand Anionic Atoms', None)

    @property
    def Cationic_Atoms(self):
        return self._parent.dict.get('Ligand Cationic Atoms', None)

    @property
    def Rule_of_Five_Violation(self):
        return self._parent.dict.get('Ligand Rule of Five Violation', None)

    @property
    def Rotatable_Bonds(self):
        return self._parent.dict.get('Ligand Rotatable Bonds', None)

    @property
    def Mass_Center_X(self):
        return self._parent.dict.get('Ligand Mass Center (X)', None)

    @property
    def Mass_Center_Y(self):
        return self._parent.dict.get('Ligand Mass Center (Y)', None)

    @property
    def Mass_Center_Z(self):
        return self._parent.dict.get('Ligand Mass Center (Z)', None)
    
    
class HTML_Protein():

    def __init__(self, parent: HTML):

        self._parent = parent

        chain_names = set(self.Chain_Names.split(',')) if isinstance(self.Chain_Names, str) else set()
        percentage_per_chain = self.Percentage_of_Residues_within_binding_site_per_Chain
        if isinstance(percentage_per_chain, str):
            site_distribution = {x.split(' = ')[0]: int(x.split(' = ')[1]) for x in percentage_per_chain.split(' / ')}
            chain_names |= set(site_distribution.keys())
        else:
            site_distribution = {}

        scopes = self.SCOPe_Chain_Classes
        if isinstance(scopes, str) and scopes:
            scopes_dict = {s.split(' ')[0]:s.split(' ')[1] for s in scopes.split(' / ')}
            chain_names |= set(scopes_dict.keys())
        else:
            scopes_dict = {}

        self._chains = []
        for name in chain_names:
            self._chains.append(
                HTML_Chain(
                    parent=self,
                    name=name,
                    percent=site_distribution.get(name, None if not site_distribution else 0),
                    scope=scopes_dict.get(name, None)
                )
            )

        self._properties = {'scPDB ID', 'PDB ID', 'PDB URL', 'Resolution', 'Method', 'Deposition Date', 'Name', 'Uniprot ID',
                            'AC', 'Organism', 'Reign', 'TaxID', 'EC', 'Percentage of Residues within binding site per Chain',
                            'Chains', 'Number of Chains', 'B-Factor of Binding Site', 'Number of Residues in Binding Site',
                            'Standard Amino Acids in Binding Site', 'Non Standard Amino Acids in Binding Site', 'Water Molecules in Binding Site',
                            'Cofactors in Binding Site', 'Metals in Binding Site', 'Cavity Ligandability', 'Cavity Volume (A^3)',
                            'Cavity % Hydrophobic', 'Cavity % Polar', 'SCOPe Chain Classes', 'Chain Site Distribution', 'Dominant Chain',
                            'EC1', 'EC2', 'EC3', 'EC4', 'EC12', 'EC123', 'SCOPe Dominant Chain', 'SCOPe1 Dominant Chain', 'SCOPe2 Dominant Chain',
                            'SCOPe3 Dominant Chain', 'SCOPe4 Dominant Chain', 'SCOPe12 Dominant Chain', 'SCOPe123 Dominant Chain'}


    def __str__(self):
        return f'<HTML Protein {self._parent._scpdb.scpdb}>'
    
    def __repr__(self):
        return str(self)
    
    @property
    def dict(self):
        return {k: self._parent.dict[k] for k in self._parent.dict if k in self._properties}
    
    @property
    def Chains(self):
        return self._chains
    
    def Get_Chain(self, name: str):

        for chain in self._chains:
            if chain.Name.lower() == name.lower():
                return chain
            
        return None

    @property
    def scPDB_ID(self):
        return self._parent.dict.get('scPDB ID', None)

    @property
    def PDB_ID(self):
        return self._parent.dict.get('PDB ID', None)

    @property
    def PDB_url(self):
        return self._parent.dict.get('PDB URL', None)

    @property
    def Resolution(self):
        return self._parent.dict.get('Resolution', None)

    @property
    def Method(self):
        return self._parent.dict.get('Method', None)

    @property
    def Deposition_Date(self):
        return self._parent.dict.get('Deposition Date', None)

    @property
    def Name(self):
        return self._parent.dict.get('Name', None)

    @property
    def Uniprot_Tag(self):
        return self._parent.dict.get('Uniprot ID', None)

    @property
    def Uniprot_ID(self):
        return self._parent.dict.get('AC', None)

    @property
    def Organism(self):
        return self._parent.dict.get('Organism', None)

    @property
    def Reign(self):
        return self._parent.dict.get('Reign', None)

    @property
    def TaxID(self):
        return self._parent.dict.get('TaxID', None)

    @property
    def EC(self):
        return self._parent.dict.get('EC', None)
    
    @property
    def EC1(self):
        return self._parent.dict.get('EC1', None)
    
    @property
    def EC2(self):
        return self._parent.dict.get('EC2', None)
    
    @property
    def EC3(self):
        return self._parent.dict.get('EC3', None)
    
    @property
    def EC4(self):
        return self._parent.dict.get('EC4', None)
    
    @property
    def EC12(self):
        return self._parent.dict.get('EC12', None)
    
    @property
    def EC123(self):
        return self._parent.dict.get('EC123', None)
    
    @property
    def SCOPe_Chain_Classes(self):
        return self._parent.dict.get('SCOPe Chain Classes', None)
    
    @property
    def Chain_Site_Distribution(self):
        return self._parent.dict.get('Chain Site Distribution', None)

    @property
    def Dominant_Chain(self):
        return self._parent.dict.get('Dominant Chain', None)

    @property
    def SCOPe_Dominant_Chain(self):
        return self._parent.dict.get('SCOPe Dominant Chain', None)

    @property
    def SCOPe1_Dominant_Chain(self):
        return self._parent.dict.get('SCOPe1 Dominant Chain', None)

    @property
    def SCOPe2_Dominant_Chain(self):
        return self._parent.dict.get('SCOPe2 Dominant Chain', None)

    @property
    def SCOPe3_Dominant_Chain(self):
        return self._parent.dict.get('SCOPe3 Dominant Chain', None)

    @property
    def SCOPe4_Dominant_Chain(self):
        return self._parent.dict.get('SCOPe4 Dominant Chain', None)

    @property
    def SCOPe12_Dominant_Chain(self):
        return self._parent.dict.get('SCOPe12 Dominant Chain', None)

    @property
    def SCOPe123_Dominant_Chain(self):
        return self._parent.dict.get('SCOPe123 Dominant Chain', None)

    @property
    def Percentage_of_Residues_within_binding_site_per_Chain(self):
        return self._parent.dict.get('Percentage of Residues within binding site per Chain', None)

    @property
    def Chain_Names(self):
        return self._parent.dict.get('Chains', None)

    @property
    def Number_of_Chains(self):
        return self._parent.dict.get('Number of Chains', None)

    @property
    def B_Factor_of_Binding_Site(self):
        return self._parent.dict.get('B-Factor of Binding Site', None)

    @property
    def Number_of_Residues_in_Binding_Site(self):
        return self._parent.dict.get('Number of Residues in Binding Site', None)

    @property
    def Standard_Amino_Acids_in_Binding_Site(self):
        return self._parent.dict.get('Standard Amino Acids in Binding Site', None)

    @property
    def Non_Standard_Amino_Acids_in_Binding_Site(self):
        return self._parent.dict.get('Non Standard Amino Acids in Binding Site', None)

    @property
    def Water_Molecules_in_Binding_Site(self):
        return self._parent.dict.get('Water Molecules in Binding Site', None)

    @property
    def Cofactors_in_Binding_Site(self):
        return self._parent.dict.get('Cofactors in Binding Site', None)

    @property
    def Metals_in_Binding_Site(self):
        return self._parent.dict.get('Metals in Binding Site', None)

    @property
    def Cavity_Ligandability(self):
        return self._parent.dict.get('Cavity Ligandability', None)

    @property
    def Cavity_Volume(self):
        return self._parent.dict.get('Cavity Volume (A^3)', None)

    @property
    def Cavity_Percent_Hydrophobic(self):
        return self._parent.dict.get('Cavity % Hydrophobic', None)

    @property
    def Cavity_Percent_Polar(self):
        return self._parent.dict.get('Cavity % Polar', None)
    

class HTML_Chain():

    def __init__(self, parent: HTML_Protein, name: str, scope: str, percent):

        self._parent = parent
        self._name = name
        self._scope = scope
        self._percent = percent
        self._scope1, self._scope2, self._scope3, self._scope4 = None, None, None, None
        self._scope12, self._scope123 = None, None
        if isinstance(self._scope, str):
            self._scope1 = self._scope.split('.')[0]
            self._scope2 = self._scope.split('.')[1]
            self._scope3 = self._scope.split('.')[2]
            self._scope4 = self._scope.split('.')[3]
            self._scope12 =  ".".join(self._scope.split('.')[0:2])
            self._scope123 = ".".join(self._scope.split('.')[0:3])
        pass

    @property
    def dict(self):
        return {'Name': self.Name,
                'scPDB ID': self._parent._parent.scPDB_ID,
                'PDB ID': self._parent._parent.PDB_ID,
                'Site Percent': self.Site_Percent,
                'SCOPe': self.SCOPe,
                'SCOPe1': self.SCOPe1,
                'SCOPe2': self.SCOPe2,
                'SCOPe3': self.SCOPe3,
                'SCOPe4': self.SCOPe4,
                'SCOPe12': self.SCOPe12,
                'SCOPe123': self.SCOPe123}

    def __str__(self):
        return f'<HTML Protein Chain {self._name} of {self._parent._parent._scpdb.scpdb}>'
    
    def __repr__(self):
        return str(self)
    
    @property
    def scPDB_ID(self):
        return self._parent._parent.scPDB_ID
    
    @property
    def PDB_ID(self):
        return self._parent._parent.PDB_ID

    @property
    def Name(self):
        return self._name
    
    @property
    def Site_Percent(self):
        return int(self._percent) if isinstance(self._percent, str) else self._percent
    
    @property
    def SCOPe(self):
        return self._scope
    
    @property
    def SCOPe1(self):
        return self._scope1
    
    @property
    def SCOPe2(self):
        return self._scope2
    
    @property
    def SCOPe3(self):
        return self._scope3
    
    @property
    def SCOPe4(self):
        return self._scope4
    
    @property
    def SCOPe12(self):
        return self._scope12
    
    @property
    def SCOPe123(self):
        return self._scope123
    

class PDB():

    def __init__(self, parent: SCPDB):

        self._scpdb = parent
        self._parent = parent

        if not self._scpdb.folder.rcsb_entry.exists:
            self._path = None
        else:
            self._path = self._scpdb.folder.rcsb_entry.path

        self._properties = {'diffrn_detector$detector', 'diffrn_radiation$pdbx_diffrn_protocol', 'diffrn_source$source', 'exptl$method',
                           'rcsb_entry_container_identifiers$pubmed_id', 'rcsb_entry_info$polymer_composition',
                           'rcsb_entry_info$selected_polymer_entity_types', 'refine$pdbx_method_to_determine_struct',
                           'struct_keywords$pdbx_keywords', 'struct_keywords$text', 
                           'exptl_crystal_grow$method', 'cell$zpdb', 'diffrn_source$pdbx_wavelength', 'exptl_crystal_grow$p_h', 
                           'rcsb_entry_info$assembly_count', 'rcsb_entry_info$branched_entity_count',
                           'rcsb_entry_info$cis_peptide_count', 'rcsb_entry_info$deposited_atom_count',
                           'rcsb_entry_info$deposited_hydrogen_atom_count', 'rcsb_entry_info$deposited_modeled_polymer_monomer_count',
                           'rcsb_entry_info$deposited_nonpolymer_entity_instance_count', 'rcsb_entry_info$deposited_polymer_entity_instance_count',
                           'rcsb_entry_info$deposited_polymer_monomer_count', 'rcsb_entry_info$deposited_solvent_atom_count',
                           'rcsb_entry_info$deposited_unmodeled_polymer_monomer_count', 'rcsb_entry_info$disulfide_bond_count',
                           'rcsb_entry_info$entity_count', 'rcsb_entry_info$inter_mol_covalent_bond_count',
                           'rcsb_entry_info$inter_mol_metalic_bond_count', 'rcsb_entry_info$nonpolymer_entity_count',
                           'rcsb_entry_info$polymer_entity_count', 'rcsb_entry_info$polymer_entity_count_dna',
                           'rcsb_entry_info$polymer_entity_count_rna', 'rcsb_entry_info$polymer_entity_count_nucleic_acid',
                           'rcsb_entry_info$polymer_entity_count_nucleic_acid_hybrid', 'rcsb_entry_info$polymer_entity_count_protein',
                           'rcsb_entry_info$polymer_monomer_count_maximum', 'rcsb_entry_info$polymer_monomer_count_minimum',
                           'rcsb_entry_info$solvent_entity_count', 'refine_hist$number_atoms_solvent', 'refine_hist$number_atoms_total',
                           'refine_hist$pdbx_number_atoms_ligand', 'refine_hist$pdbx_number_atoms_nucleic_acid',
                           'refine_hist$pdbx_number_atoms_protein',
                           'cell$length_a', 'cell$length_b', 'cell$length_c', 'diffrn$ambient_temp', 'rcsb_entry_info$molecular_weight',
                           'rcsb_entry_info$nonpolymer_molecular_weight_maximum', 'rcsb_entry_info$nonpolymer_molecular_weight_minimum',
                           'rcsb_entry_info$polymer_molecular_weight_maximum', 'rcsb_entry_info$polymer_molecular_weight_minimum',
                           'rcsb_entry_info$resolution_combined'}

        self._dict = self._scpdb.folder.rcsb_entry.content
        if self._dict:
            self._dict = self._json_to_dict(self._dict)
            self._dict = {k: self._dict[k] for k in self._dict if k in self._properties}
        else:
            self._dict = {}
        
        # Link to mol2
        self._main_ligand = self._scpdb.mol2.protein.raw
        self._main_ligand = self._main_ligand.split('\n')[1].split('_')[1]
        self._dominant_chain = self._scpdb.mol2.protein.Dominant_Chain
        
        entities = self._scpdb.folder.rcsb_entities
        self._entities = []
        self._protein_entities = []
        self._polymer_entities = []
        self._na_entities = []
        self._dna_entities = []
        self._rna_entities = []
        self._na_hybrid_entities = []
        self._nonpolymer_entities = []
        self._branched_entities = []
        self._chains = {}
        self._dominant_chain_rcsb = None
        self._main_ligand_rcsb = None
        for entity in entities:
            entity_js = entity.content
            if "entity_poly" in entity_js.keys():
                group = "polymer"
                if entity_js["entity_poly"]["rcsb_entity_polymer_type"].lower() == 'protein':
                    group = "protein"
                elif entity_js["entity_poly"]["rcsb_entity_polymer_type"].lower() == 'dna':
                    group = "dna"
                elif entity_js["entity_poly"]["rcsb_entity_polymer_type"].lower() == 'rna':
                    group = "dna"
                elif entity_js["entity_poly"]["rcsb_entity_polymer_type"].lower() == 'na-hybrid':
                    group = "na-hybrid"
                else:
                    raise KeyError(f'unknown polymer type: {entity_js["entity_poly"]["rcsb_entity_polymer_type"]}')

            elif "pdbx_entity_nonpoly" in entity_js.keys():
                group = "nonpolymer"
            elif "pdbx_entity_branch" in  entity_js.keys():
                group = "branched"

            else:
                raise KeyError(f'unknown entity group: {entity.path}')
            
            if group == "protein":
                entity_object = RCSB_Protein(self, entity)
                if entity_object.Strands is not None and self._dominant_chain is not None and self._dominant_chain.Name in entity_object.Strands.split(','):
                    self._dominant_chain_rcsb = entity_object
            elif group in ("dna", "rna", "na-hybrid"):
                entity_object = RCSB_NA(self, entity)
                if entity_object.SubType is not None and self._main_ligand.lower() == entity_object.SubType.lower():
                    self._main_ligand_rcsb = entity_object
            
            elif group == "nonpolymer":
                entity_object = RCSB_Ligand(self, entity)
                if entity_object.Compound_Id is not None and self._main_ligand.lower() == entity_object.Compound_Id.lower():
                    self._main_ligand_rcsb = entity_object
            else:
                entity_object = RCSB_Branched(self, entity)
                
            self._entities.append(entity_object)
            if group == "protein":
                self._protein_entities.append(entity_object)
                self._polymer_entities.append(entity_object)
                if entity_object.Strands is not None:
                    chain_names = entity_object.Strands.split(',')
                    for name in chain_names:
                        self._chains.update({name: entity_object})
            elif group in ("dna", "rna", "na-hybrid"):
                self._na_entities.append(entity_object)
                self._polymer_entities.append(entity_object)
                if group == "dna":
                    self._dna_entities.append(entity_object)
                elif group == "rna":
                    self._rna_entities.append(entity_object)
                else:
                    self._na_hybrid_entities.append(entity_object)
            elif group == "nonpolymer":
                self._nonpolymer_entities.append(entity_object)
            else:
                self._branched_entities.append(entity_object)

        self._dict.update({'Dominant Chain': self._dominant_chain.Name if self._dominant_chain is not None else None,
                           'Main Ligand': self._main_ligand,
                           'Dominant Chain Found': self.Dominant_Chain is not None,
                           'Main Ligand Found': self.Main_Ligand is not None,
                           'Number of Chains': self.Number_Of_Chains,
                           'Number of Different Chains': self.Number_Of_Different_Chains,
                           'Chain Configuration': self.Chain_Configuration,
                           'Chain Names': self.Chain_Names,
                           'Number of Entities': self.Number_Of_Entities,
                           'Number of Polymers': self.Number_Of_Polymers,
                           'Number of Non Polymers': self.Number_Of_NonPolymers,
                           'Number of Proteins': self.Number_Of_Proteins,
                           'Number of Branched': self.Number_Of_Branched,
                           'Number of DNAs': self.Number_Of_DNAs,
                           'Number of RNAs': self.Number_Of_RNAs,
                           'Number of NAs': self.Number_Of_NAs,
                           'Number of NA Hybrids': self.Number_Of_NA_Hybrids})
        
        self._prefix = 'rcsb$'

        
    def _json_to_dict(self, entry_js: dict | None):

        if entry_js:
            rcsb_entry_dict = {}

            for key in entry_js:
                if 'audit' in key or 'citation' in key or 'revision' in key or 'refine_ls_restr' in key or 'software' in key or 'rcsb_binding_affinity' in key:
                    continue
                if isinstance(entry_js[key], list):
                    block = entry_js[key][0]
                    for k in block:
                        if 'details' in k:
                            continue
                        if isinstance(block[k], list):
                            rcsb_entry_dict.update({f'{key}${k}': ';'.join(str(v) for v in block[k])})
                        else:
                            rcsb_entry_dict.update({f'{key}${k}': block[k]})

                elif isinstance(entry_js[key], dict):

                    block = entry_js[key]
                    for k in block:
                        if 'details' in k:
                            continue
                        if isinstance(block[k], list):
                            rcsb_entry_dict.update({f'{key}${k}': ';'.join(str(v) for v in block[k])})
                        else:
                            rcsb_entry_dict.update({f'{key}${k}': block[k]})
                else:
                    rcsb_entry_dict.update({f'{key}': entry_js[key]})
        
        return rcsb_entry_dict
    
    def __str__(self):
        return f'<RCSB Entry {self._scpdb.scpdb}>'
    
    def __repr__(self):
        return str(self)

    @property
    def exists(self):
        return self._scpdb.folder.rcsb_entry.exists
    
    @property
    def path(self):
        return self._scpdb.folder.rcsb_entry.path

    @property
    def dict(self):
        return self._dict
    
    @property
    def dict_with_prefix(self):
        return {self._prefix + k: self._dict[k] for k in self._dict}
    
    @property
    def dict_complete(self):
        
        _dict = {}
        _dict.update(self.dict_with_prefix)
        if self.Dominant_Chain is not None:
            _dict.update(self.Dominant_Chain.dict_with_prefix)
        if self.Main_Ligand is not None:
            _dict.update(self.Main_Ligand.dict_with_prefix)

        return _dict
    
    @property
    def scPDB_ID(self):
        return self._scpdb.scpdb
    
    @property
    def PDB_ID(self):
        return self._scpdb.pdb
    
    @property
    def Dominant_Chain(self):
        return self._dominant_chain_rcsb
    
    @property
    def Chain_Names(self):
        return sorted(self._chains.keys())
    
    @property
    def Number_Of_Chains(self):
        return len(self._chains)
    
    @property
    def Number_Of_Different_Chains(self):
        return len(set(v.FASTAcan for v in self._chains.values()))
    
    @property
    def Chain_Configuration(self):
        return '+'.join(str(x) for x in sorted(Counter(v.FASTAcan for v in self._chains.values()).values(), reverse=True))
    
    def Get_Chain(self, name):
        return self._chains.get(name.upper(), None)
    
    @property
    def Main_Ligand(self):
        return self._main_ligand_rcsb
    
    @property
    def Entities(self):
        return self._entities
    
    @property
    def Number_Of_Entities(self):
        return len(self._entities)
    
    @property
    def Number_Of_Polymers(self):
        return len(self._polymer_entities)
    
    @property
    def Number_Of_NonPolymers(self):
        return len(self._nonpolymer_entities)
    
    @property
    def Number_Of_Branched(self):
        return len(self._branched_entities)
    
    @property
    def Number_Of_Proteins(self):
        return len(self._protein_entities)
    
    @property
    def Number_Of_NAs(self):
        return len(self._na_entities)
    
    @property
    def Number_Of_DNAs(self):
        return len(self._dna_entities)
    
    @property
    def Number_Of_RNAs(self):
        return len(self._rna_entities)
    
    @property
    def Number_Of_NA_Hybrids(self):
        return len(self._na_hybrid_entities)
    
    @property
    def Protein_Entities(self):
        return self._protein_entities
    
    @property
    def Polymer_Entities(self):
        return self._polymer_entities
    
    @property
    def NA_Entities(self):
        return self._na_entities
    
    @property
    def Nonpolymer_Entities(self):
        return self._nonpolymer_entities
    
    @property
    def Branched_Entities(self):
        return self._branched_entities
    
    def Get_Entity(self, number: str | int):

        for e in self._entities:
            if str(e.Entity_ID) == str(number):
                return e
        return None
    
    @property
    def Polymer_Monomer_Count_Minimum(self):
        return self._parent.dict.get('rcsb_entry_info$polymer_monomer_count_minimum', None)

    @property
    def Disulfide_Bond_Count(self):
        return self._parent.dict.get('rcsb_entry_info$disulfide_bond_count', None)

    @property
    def Method(self):
        return self._parent.dict.get('exptl$method', None)

    @property
    def Crystal_Grow_Method(self):
        return self._parent.dict.get('exptl_crystal_grow$method', None)

    @property
    def Diffaction_Source(self):
        return self._parent.dict.get('diffrn_source$source', None)

    @property
    def Zpdb(self):
        return self._parent.dict.get('cell$zpdb', None)

    @property
    def Deposited_Atom_Count(self):
        return self._parent.dict.get('rcsb_entry_info$deposited_atom_count', None)

    @property
    def Deposited_Polymer_Entity_Instance_Count(self):
        return self._parent.dict.get('rcsb_entry_info$deposited_polymer_entity_instance_count', None)

    @property
    def Deposited_Unmodeled_Polymer_Monomer_Count(self):
        return self._parent.dict.get('rcsb_entry_info$deposited_unmodeled_polymer_monomer_count', None)

    @property
    def Pubmed_ID(self):
        return self._parent.dict.get('rcsb_entry_container_identifiers$pubmed_id', None)

    @property
    def Molecular_Weight(self):
        return self._parent.dict.get('rcsb_entry_info$molecular_weight', None)

    @property
    def Text(self):
        return self._parent.dict.get('struct_keywords$text', None)

    @property
    def Deposited_Solvent_Atom_Count(self):
        return self._parent.dict.get('rcsb_entry_info$deposited_solvent_atom_count', None)

    @property
    def Crystal_Grow_PH(self):
        return self._parent.dict.get('exptl_crystal_grow$p_h', None)

    @property
    def Polymer_Entity_Count_Nucleic_Acid_Hybrid(self):
        return self._parent.dict.get('rcsb_entry_info$polymer_entity_count_nucleic_acid_hybrid', None)

    @property
    def Number_Atoms_Nucleic_Acid(self):
        return self._parent.dict.get('refine_hist$pdbx_number_atoms_nucleic_acid', None)

    @property
    def Nonpolymer_Molecular_Weight_Maximum(self):
        return self._parent.dict.get('rcsb_entry_info$nonpolymer_molecular_weight_maximum', None)

    @property
    def Polymer_Entity_Count(self):
        return self._parent.dict.get('rcsb_entry_info$polymer_entity_count', None)

    @property
    def Pdbx_Keywords(self):
        return self._parent.dict.get('struct_keywords$pdbx_keywords', None)

    @property
    def Deposited_Hydrogen_Atom_Count(self):
        return self._parent.dict.get('rcsb_entry_info$deposited_hydrogen_atom_count', None)

    @property
    def Inter_Mol_Covalent_Bond_Count(self):
        return self._parent.dict.get('rcsb_entry_info$inter_mol_covalent_bond_count', None)

    @property
    def Polymer_Entity_Count_Protein(self):
        return self._parent.dict.get('rcsb_entry_info$polymer_entity_count_protein', None)

    @property
    def Polymer_Molecular_Weight_Minimum(self):
        return self._parent.dict.get('rcsb_entry_info$polymer_molecular_weight_minimum', None)

    @property
    def Polymer_Entity_Count_Nucleic_Acid(self):
        return self._parent.dict.get('rcsb_entry_info$polymer_entity_count_nucleic_acid', None)

    @property
    def Resolution_Combined(self):
        return self._parent.dict.get('rcsb_entry_info$resolution_combined', None)

    @property
    def Polymer_Monomer_Count_Maximum(self):
        return self._parent.dict.get('rcsb_entry_info$polymer_monomer_count_maximum', None)

    @property
    def Length_A(self):
        return self._parent.dict.get('cell$length_a', None)

    @property
    def Branched_Entity_Count(self):
        return self._parent.dict.get('rcsb_entry_info$branched_entity_count', None)

    @property
    def Polymer_Entity_Count_RNA(self):
        return self._parent.dict.get('rcsb_entry_info$polymer_entity_count_rna', None)

    @property
    def Diffraction_Ambient_Temp(self):
        return self._parent.dict.get('diffrn$ambient_temp', None)

    @property
    def Length_C(self):
        return self._parent.dict.get('cell$length_c', None)

    @property
    def Polymer_Molecular_Weight_Maximum(self):
        return self._parent.dict.get('rcsb_entry_info$polymer_molecular_weight_maximum', None)

    @property
    def Diffraction_Wavelength(self):
        return self._parent.dict.get('diffrn_source$pdbx_wavelength', None)

    @property
    def Number_Atoms_Protein(self):
        return self._parent.dict.get('refine_hist$pdbx_number_atoms_protein', None)

    @property
    def Number_Atoms_Total(self):
        return self._parent.dict.get('refine_hist$number_atoms_total', None)

    @property
    def Detector(self):
        return self._parent.dict.get('diffrn_detector$detector', None)

    @property
    def Deposited_Nonpolymer_Entity_Instance_Count(self):
        return self._parent.dict.get('rcsb_entry_info$deposited_nonpolymer_entity_instance_count', None)

    @property
    def Inter_Mol_Metalic_Bond_Count(self):
        return self._parent.dict.get('rcsb_entry_info$inter_mol_metalic_bond_count', None)

    @property
    def Solvent_Entity_Count(self):
        return self._parent.dict.get('rcsb_entry_info$solvent_entity_count', None)

    @property
    def Polymer_Composition(self):
        return self._parent.dict.get('rcsb_entry_info$polymer_composition', None)

    @property
    def Assembly_Count(self):
        return self._parent.dict.get('rcsb_entry_info$assembly_count', None)

    @property
    def Cis_Peptide_Count(self):
        return self._parent.dict.get('rcsb_entry_info$cis_peptide_count', None)

    @property
    def Selected_Polymer_Entity_Types(self):
        return self._parent.dict.get('rcsb_entry_info$selected_polymer_entity_types', None)

    @property
    def Method_To_Determine_Struct(self):
        return self._parent.dict.get('refine$pdbx_method_to_determine_struct', None)

    @property
    def Nonpolymer_Molecular_Weight_Minimum(self):
        return self._parent.dict.get('rcsb_entry_info$nonpolymer_molecular_weight_minimum', None)

    @property
    def Entity_Count(self):
        return self._parent.dict.get('rcsb_entry_info$entity_count', None)

    @property
    def Length_B(self):
        return self._parent.dict.get('cell$length_b', None)

    @property
    def Deposited_Modeled_Polymer_Monomer_Count(self):
        return self._parent.dict.get('rcsb_entry_info$deposited_modeled_polymer_monomer_count', None)

    @property
    def Number_Atoms_Ligand(self):
        return self._parent.dict.get('refine_hist$pdbx_number_atoms_ligand', None)

    @property
    def Nonpolymer_Entity_Count(self):
        return self._parent.dict.get('rcsb_entry_info$nonpolymer_entity_count', None)

    @property
    def Polymer_Entity_Count_DNA(self):
        return self._parent.dict.get('rcsb_entry_info$polymer_entity_count_dna', None)

    @property
    def Diffraction_Protocol(self):
        return self._parent.dict.get('diffrn_radiation$pdbx_diffrn_protocol', None)

    @property
    def Number_Atoms_Solvent(self):
        return self._parent.dict.get('refine_hist$number_atoms_solvent', None)

    @property
    def Deposited_Polymer_Monomer_Count(self):
        return self._parent.dict.get('rcsb_entry_info$deposited_polymer_monomer_count', None)
    

class RCSB_Protein():

    def __init__(self, parent: HTML, path: Path):

        self._parent = parent
        self._path_object = path
        self._entity_id = int(self._path_object.tail.replace('.json', '').split('_')[-1])


        self._properties = {'entity_poly$pdbx_seq_one_letter_code', 'entity_poly$pdbx_seq_one_letter_code_can', 'entity_poly$pdbx_strand_id',
                            'entity_src_gen$gene_src_genus', 'entity_src_gen$host_org_genus', 'entity_src_gen$pdbx_gene_src_scientific_name',
                            'entity_src_gen$pdbx_host_org_scientific_name', 'entity_poly$rcsb_non_std_monomers', 'entity_poly$rcsb_mutation_count',
                            'entity_poly$rcsb_non_std_monomer_count', 'rcsb_polymer_entity$pdbx_number_of_molecules', 'rcsb_polymer_entity$formula_weight',
                            'entity_poly$rcsb_sample_sequence_length'}
        
        self._dict = self._path_object.content
        if self._dict:
            self._dict = self._parent._json_to_dict(self._dict)
            self._dict = {k: self._dict[k].upper().strip() if isinstance(self._dict[k], str) else self._dict[k] for k in self._dict if k in self._properties}

        if not self._dict:
            self._dict = {}
        
        self._dict.update({'scPDB ID': self.scPDB_ID, 'PDB ID': self.PDB_ID, 'Entity Type': self.Type, 'Entity ID': self.ID, 'Uniprot ID': self.UniProt_ID})
        self._prefix = 'rcsb_prot$'


    def __str__(self):
        return f'<RCSB Protein Entity {self._parent._scpdb.scpdb}>'
    
    def __repr__(self):
        return str(self)
    
    @property
    def path(self):
        return self._path_object.path
    
    @property
    def Type(self):
        return "polymer protein"
    
    @property
    def ID(self):
        return self._entity_id
    
    @property
    def UniProt_ID(self):
        found = re.findall(r'([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})', str(self._path_object.content))
        if not found:
            return None
        return sorted(filter(lambda x: x, sorted(filter(lambda x: x, set(found)))[0]))[0]
    
    @property
    def dict(self):
        return self._dict
    
    @property
    def dict_with_prefix(self):
        return {self._prefix + k: self._dict[k] for k in self._dict}

    @property
    def scPDB_ID(self):
        return self._parent._scpdb.scpdb

    @property
    def PDB_ID(self):
        return self._parent._scpdb.pdb
    
    @property
    def SourceGenus(self):
        return self.dict.get('entity_src_gen$gene_src_genus', None)

    @property
    def FASTA(self):
        return self.dict.get('entity_poly$pdbx_seq_one_letter_code', None)

    @property
    def Mutation_Count(self):
        return self.dict.get('entity_poly$rcsb_mutation_count', None)

    @property
    def Non_Standard_Monomers(self):
        return self.dict.get('entity_poly$rcsb_non_std_monomers', None)

    @property
    def Source_Scientific_Name(self):
        return self.dict.get('entity_src_gen$pdbx_gene_src_scientific_name', None)

    @property
    def Number_Of_Molecules(self):
        return self.dict.get('rcsb_polymer_entity$pdbx_number_of_molecules', None)

    @property
    def FASTAcan(self):
        return self.dict.get('entity_poly$pdbx_seq_one_letter_code_can', None)

    @property
    def Formula_Weight(self):
        return self.dict.get('rcsb_polymer_entity$formula_weight', None)

    @property
    def Non_Standard_Monomer_Count(self):
        return self.dict.get('entity_poly$rcsb_non_std_monomer_count', None)

    @property
    def Host_Organism_Genus(self):
        return self.dict.get('entity_src_gen$host_org_genus', None)

    @property
    def Strands(self):
        return self.dict.get('entity_poly$pdbx_strand_id', None)
    
    @property
    def Length(self):
        return self.dict.get('entity_poly$rcsb_sample_sequence_length', None)

    @property
    def Host_Organism_Scientific_Name(self):
        return self.dict.get('entity_src_gen$pdbx_host_org_scientific_name', None)
    

class RCSB_Ligand():

    def __init__(self, parent: HTML, path: Path):

        self._parent = parent
        self._path_object = path
        self._entity_id = int(self._path_object.tail.replace('.json', '').split('_')[-1])


        self._properties = {'pdbx_entity_nonpoly$comp_id', 'pdbx_entity_nonpoly$name', 
                           'rcsb_nonpolymer_entity$pdbx_number_of_molecules', 'rcsb_nonpolymer_entity$formula_weight'
                            }
        
        self._dict = self._path_object.content
        if self._dict:
            self._dict = self._parent._json_to_dict(self._dict)
            self._dict = {k: self._dict[k].upper().strip() if isinstance(self._dict[k], str) else self._dict[k] for k in self._dict if k in self._properties}

        if not self._dict:
            self._dict = {}
        
        self._dict.update({'scPDB ID': self.scPDB_ID, 'PDB ID': self.PDB_ID, 'Entity Type': self.Type, 'Entity ID': self.ID})
        self._prefix = 'rcsb_ligand$'


    def __str__(self):
        return f'<RCSB Nonpolymer Entity {self._parent._scpdb.scpdb}>'
    
    def __repr__(self):
        return str(self)
    
    @property
    def path(self):
        return self._path_object.path
    
    @property
    def Type(self):
        return "nonpolymer ligand"
    
    @property
    def ID(self):
        return self._entity_id
    
    @property
    def dict(self):
        return self._dict
    
    @property
    def dict_with_prefix(self):
        return {self._prefix + k: self._dict[k] for k in self._dict}

    @property
    def scPDB_ID(self):
        return self._parent._scpdb.scpdb

    @property
    def PDB_ID(self):
        return self._parent._scpdb.pdb
    
    @property
    def Compound_Id(self):
        return self.dict.get('pdbx_entity_nonpoly$comp_id', None)

    @property
    def Formula_Weight(self):
        return self.dict.get('rcsb_nonpolymer_entity$formula_weight', None)

    @property
    def Name(self):
        return self.dict.get('pdbx_entity_nonpoly$name', None)

    @property
    def Number_Of_Molecules(self):
        return self.dict.get('rcsb_nonpolymer_entity$pdbx_number_of_molecules', None)


class RCSB_NA():

    def __init__(self, parent: HTML, path: Path):

        self._parent = parent
        self._path_object = path
        self._entity_id = int(self._path_object.tail.replace('.json', '').split('_')[-1])


        self._properties = {'entity_poly$pdbx_seq_one_letter_code', 'entity_poly$pdbx_seq_one_letter_code_can', 'entity_poly$pdbx_strand_id',
                            'entity_poly$rcsb_artifact_monomer_count', 'entity_poly$rcsb_conflict_count', 'entity_poly$rcsb_deletion_count',
                            'entity_poly$rcsb_entity_polymer_type', 'entity_poly$rcsb_insertion_count', 'entity_poly$rcsb_mutation_count',
                            'entity_poly$rcsb_non_std_monomer_count', 'entity_poly$rcsb_sample_sequence_length', 'entity_poly$type',
                            'rcsb_polymer_entity$formula_weight', 'rcsb_polymer_entity$pdbx_description', 'rcsb_polymer_entity$pdbx_number_of_molecules',
                            'rcsb_polymer_entity_container_identifiers$chem_comp_monomers'}
        
        self._dict = self._path_object.content
        if self._dict:
            self._dict = self._parent._json_to_dict(self._dict)
            self._dict = {k: self._dict[k].upper().strip() if isinstance(self._dict[k], str) else self._dict[k] for k in self._dict if k in self._properties}

        if not self._dict:
            self._dict = {}
        
        self._dict.update({'scPDB ID': self.scPDB_ID, 'PDB ID': self.PDB_ID, 'Entity Type': self.Type, 'Entity ID': self.ID})
        self._prefix = 'rcsb_na$'


    def __str__(self):
        return f'<RCSB NA Entity {self._parent._scpdb.scpdb}>'
    
    def __repr__(self):
        return str(self)
    
    @property
    def path(self):
        return self._path_object.path
    
    @property
    def ID(self):
        return self._entity_id
    
    @property
    def dict(self):
        return self._dict
    
    @property
    def dict_with_prefix(self):
        return {self._prefix + k: self._dict[k] for k in self._dict}

    @property
    def scPDB_ID(self):
        return self._parent._scpdb.scpdb

    @property
    def PDB_ID(self):
        return self._parent._scpdb.pdb
    
    @property
    def Strands(self):
        return self.dict.get('entity_poly$pdbx_seq_one_letter_code_canentity_poly$pdbx_strand_id', None)

    @property
    def Type(self):
        return "polymer nucleic acid" + (f' {self.SubType}'.lower() if self.SubType else ' unknown')
        
    @property
    def SubType(self):

        return self.dict.get('entity_poly$rcsb_entity_polymer_type', None)

    @property
    def Deletion_Count(self):
        return self.dict.get('entity_poly$rcsb_deletion_count', None)

    @property
    def Artifact_Monomer_Count(self):
        return self.dict.get('entity_poly$rcsb_artifact_monomer_count', None)

    @property
    def Non_Standard_Monomer_Count(self):
        return self.dict.get('entity_poly$rcsb_non_std_monomer_count', None)

    @property
    def Mutation_Count(self):
        return self.dict.get('entity_poly$rcsb_mutation_count', None)

    @property
    def FASTA(self):
        return self.dict.get('entity_poly$"pdbx_seq_one_letter_code', None)

    @property
    def Description(self):
        return self.dict.get('rcsb_polymer_entity$pdbx_description', None)

    @property
    def Insertion_Count(self):
        return self.dict.get('entity_poly$rcsb_insertion_count', None)

    @property
    def Conflict_Count(self):
        return self.dict.get('entity_poly$rcsb_conflict_count', None)

    @property
    def Number_Of_Molecules(self):
        return self.dict.get('rcsb_polymer_entity$pdbx_number_of_molecules', None)

    @property
    def Formula_Weight(self):
        return self.dict.get('rcsb_polymer_entity$formula_weight', None)

    @property
    def Length(self):
        return self.dict.get('entity_poly$rcsb_sample_sequence_length', None)

    @property
    def SubType_Long(self):
        return self.dict.get('entity_poly$type', None)

    @property
    def Monomers(self):
        return self.dict.get('rcsb_polymer_entity_container_identifiers$chem_comp_monomers', None)


class RCSB_Branched():

    def __init__(self, parent: HTML, path: Path):

        self._parent = parent
        self._path_object = path
        self._entity_id = int(self._path_object.tail.replace('.json', '').split('_')[-1])


        self._properties = {'pdbx_entity_branch$rcsb_branched_component_count', 'pdbx_entity_branch$type',
                            'pdbx_entity_branch_descriptor$descriptor', 'pdbx_entity_branch_descriptor$type',
                            'rcsb_branched_entity$formula_weight', 'rcsb_branched_entity$pdbx_description',
                            'rcsb_branched_entity$pdbx_number_of_molecules',
                            'rcsb_branched_entity_container_identifiers$chem_comp_monomers'
                            }
        
        self._dict = self._path_object.content
        if self._dict:
            self._dict = self._parent._json_to_dict(self._dict)
            self._dict = {k: self._dict[k].upper().strip() if isinstance(self._dict[k], str) else self._dict[k] for k in self._dict if k in self._properties}

        if not self._dict:
            self._dict = {}
        
        self._dict.update({'scPDB ID': self.scPDB_ID, 'PDB ID': self.PDB_ID, 'Entity Type': self.Type, 'Entity ID': self.ID})
        self._prefix = 'rcsb_branched$'


    def __str__(self):
        return f'<RCSB Branched Entity {self._parent._scpdb.scpdb}>'
    
    def __repr__(self):
        return str(self)
    
    @property
    def path(self):
        return self._path_object.path
    
    @property
    def Type(self):
        return "branched" + (f' {self.SubType}'.lower() if self.SubType else ' unknown')
    
    @property
    def ID(self):
        return self._entity_id
    
    @property
    def dict(self):
        return self._dict
    
    @property
    def dict_with_prefix(self):
        return {self._prefix + k: self._dict[k] for k in self._dict}

    @property
    def scPDB_ID(self):
        return self._parent._scpdb.scpdb

    @property
    def PDB_ID(self):
        return self._parent._scpdb.pdb
    
    @property
    def Component_Count(self):
        return self.dict.get('pdbx_entity_branch$rcsb_branched_component_count', None)

    @property
    def SubType(self):
        return self.dict.get('pdbx_entity_branch$type', None)

    @property
    def Formula_Weight(self):
        return self.dict.get('rcsb_branched_entity$formula_weight', None)

    @property
    def Description(self):
        return self.dict.get('rcsb_branched_entity$pdbx_description', None)

    @property
    def Descriptor(self):
        return self.dict.get('pdbx_entity_branch_descriptor$descriptor', None)

    @property
    def Monomers(self):
        return self.dict.get('rcsb_branched_entity_container_identifiers$chem_comp_monomers', None)

    @property
    def SubType_Long(self):
        return self.dict.get('pdbx_entity_branch_descriptor$type', None)

    @property
    def Number_Of_Molecules(self):
        return self.dict.get('rcsb_branched_entity$pdbx_number_of_molecules', None)


class Mol2():

    def __init__(self, parent: SCPDB):

        self._scpdb = parent

        self._protein_path = self._scpdb.folder.protein_mol2
        self._site_path = self._scpdb.folder.site_mol2
        self._ligand_path = self._scpdb.folder.ligand_mol2
        self._cavity_path = self._scpdb.folder.cavity_mol2

        self._dict = {'Protein Path': self._protein_path.path,
                      'Site Path': self._site_path.path,
                      'Ligand Path': self._ligand_path.path,
                      'Cavity Path': self._cavity_path.path,
                      'Derived Ligand': self.Ligand_Code}
        
        self._ligand = Mol2_File(self, self._ligand_path, 'Ligand')
        self._site = Mol2_File(self, self._site_path, 'Site')
        self._protein = Mol2_File(self, self._protein_path, 'Protein')

        self._prefix = 'mol2$'
    
    def __str__(self):
        return f'<Mol2 Overview {self._scpdb.scpdb}>'
    
    def __repr__(self):
        return str(self)
    
    @property
    def Ligand_Code(self):
        return self._scpdb.folder.protein_mol2.content.split('\n')[1].split('_')[1]

    @property
    def dict(self):
        return self._dict
    
    @property
    def protein(self):
        return self._protein
    
    @property
    def site(self):
        return self._site
    
    @property
    def ligand(self):
        return self._ligand
    
    @property
    def dict_with_prefix(self):

        _dict = self._dict
        _dict.update(self.protein.dict)
        _dict.update(self.site.dict)
        _dict.update(self.ligand.dict)

        _dict = {self._prefix + k: _dict[k] for k in _dict}
        return _dict

    
    def _get_block(self, file: str, block: str):

        if file is None:
            return None

        bb = file.split('@<TRIPOS>')
        
        for b in bb:
            title = b.split('\n')[0]

            if title.lower() == block.lower():
                return list(filter(lambda line: line, b.split('\n')[1:]))

        return None

    
    def _block_to_df(self, block: list[str]):

        if block is None:
            return None

        table = [
            list(filter(lambda x: x, line.split(' ')))
            for line in block
        ]

        if not table:
            return pd.DataFrame()

        max_length = max(len(l) for l in table)
        for i in range(len(table)):
            if len(table[i]) < max_length:
                table[i].extend([None] * (max_length - len(table[i])))

        df = pd.DataFrame(table)
        return df
    
    def _get_atoms(self, raw_df: pd.DataFrame):

        if raw_df is None:
            return None
        
        if len(raw_df) == 0:
            raw_df = pd.DataFrame(columns=['Atom ID', 'TRIPOS', 'X', 'Y', 'Z', 'Type', 'Substructure ID', 'Substructure Code', 'Atom Charge'])

        raw_df.columns = ['Atom ID', 'TRIPOS', 'X', 'Y', 'Z', 'Type', 'Substructure ID', 'Substructure Code', 'Atom Charge']
        raw_df['Element'] = raw_df['Type'].map(lambda x: x.split('.')[0] if isinstance(x, str) else None)
        raw_df['Substructure Name'] = raw_df['Substructure Code'].map(
            lambda x: ''.join(xx for xx in x if xx.isalpha())
            )
        raw_df['Substructure Position'] = raw_df['Substructure Code'].map(
            lambda x: ''.join(xx for xx in x if xx.isnumeric())
            )
        
        def substructure_types(s: str):
            if s in self._scpdb.RESIDUES:
                return 'AA'
            if s == 'HOH':
                return 'WATER'
            if s in [mw.upper() for mw in list(self._scpdb.MW.keys())]:
                return 'HETERO'
            if not s:
                return None
            return 'OTHER'
        
        raw_df['Substructure Type'] = raw_df['Substructure Name'].map(lambda x: substructure_types(x))

        raw_df['Atom ID'] = raw_df['Atom ID'].astype(int)
        raw_df['X'] = raw_df['X'].astype(float)
        raw_df['Y'] = raw_df['Y'].astype(float)
        raw_df['Z'] = raw_df['Z'].astype(float)
        raw_df['Substructure ID'] = raw_df['Substructure ID'].astype(int)
        raw_df['Atom Charge'] = raw_df['Atom Charge'].astype(float)
        
        return raw_df


    def _get_substructures(self, raw_df: pd.DataFrame):

        if raw_df is None:
            return None
        
        if len(raw_df) == 0:
            raw_df = pd.DataFrame(columns=['Substructure ID', 'Substructure Code', 'Start Atom', '@1', '@2', 'Chain'])

        raw_df = raw_df.iloc[:, :6]
        raw_df.columns = ['Substructure ID', 'Substructure Code', 'Start Atom', '@1', '@2', 'Chain']
        raw_df['Substructure Position'] = raw_df['Substructure Code'].map(lambda x: x[3:])
        raw_df = raw_df.drop(columns=['@1', '@2'])

        raw_df['Substructure Name'] = raw_df['Substructure Code'].map(
            lambda x: ''.join(xx for xx in x if xx.isalpha()) if ''.join(xx for xx in x if xx.isalpha()) else x[:3]
            )
        raw_df['Substructure Position'] = raw_df['Substructure Code'].map(
            lambda x: ''.join(xx for xx in x if xx.isnumeric()) if len(''.join(xx for xx in x if xx.isnumeric())) <= 4 else x[3:]
            )
        
        def substructure_types(s: str):
            if s in self._scpdb.RESIDUES:
                return 'AA'
            if s == 'HOH':
                return 'WATER'
            if s in [mw.upper() for mw in list(self._scpdb.MW.keys())]:
                return 'HETERO'
            if not s:
                return None
            return 'OTHER'
        
        raw_df['Substructure Type'] = raw_df['Substructure Name'].map(lambda x: substructure_types(x))

        raw_df['Substructure ID'] = raw_df['Substructure ID'].astype(int)
        raw_df['Start Atom'] = raw_df['Start Atom'].astype(int)
        raw_df['Substructure Position'] = raw_df['Substructure Position'].astype(int)
        
        return raw_df


    def _get_bonds(self, raw_df: pd.DataFrame):

        if raw_df is None:
            return None
        
        if len(raw_df) == 0:
            raw_df = pd.DataFrame(columns=['Bond ID', 'Atom ID 1', 'Atom ID 2', 'Bond Type'])

        raw_df.columns = ['Bond ID', 'Atom ID 1', 'Atom ID 2', 'Bond Type']

        raw_df['Bond ID'] = raw_df['Bond ID'].astype(int)
        raw_df['Atom ID 1'] = raw_df['Atom ID 1'].astype(int)
        raw_df['Atom ID 1'] = raw_df['Atom ID 1'].astype(int)

        return raw_df
    
    def _residue_rotation_matrix(self, ca_x, ca_y, ca_z, n_x, n_y, n_z, c_x, c_y, c_z):
        """
        Compute local residue rotation matrix from CA, N, C coordinates.
        Returns
        -------
        R : numpy.ndarray (3,3)
            Rotation matrix that transforms global coordinates
            into the residue-local frame
        """

        ca = np.asarray([ca_x, ca_y, ca_z], dtype=float)
        n = np.asarray([n_x, n_y, n_z], dtype=float)
        c = np.asarray([c_x, c_y, c_z], dtype=float)

        # First axis: CA -> C
        e1 = c - ca
        e1 = e1 / np.linalg.norm(e1)

        # Vector CA -> N
        v = n - ca

        # Third axis: normal to backbone plane
        e3 = np.cross(e1, v)
        e3 = e3 / np.linalg.norm(e3)

        # Second axis
        e2 = np.cross(e3, e1)

        # Rotation matrix
        R = np.vstack([e1, e2, e3])

        return R
    
    def Save_FASTA(self):
        p = os.path.join(self._scpdb.folder_path, 'FASTA.txt')

        fasta_dict_protein = self.protein.FASTA
        length = len(list(fasta_dict_protein.values())[0])

        fasta_protein = '\n'.join(f'> {k}' + '\n' + v for k, v in fasta_dict_protein.items())
        with open(p, 'w') as f:
            f.write(fasta_protein)
        return
    
    def Save_FASTAsite(self):
        p = os.path.join(self._scpdb.folder_path, 'FASTAsite.txt')

        fasta_dict_protein = self.protein.FASTA
        fasta_dict_site = self.site.FASTA
        length = len(list(fasta_dict_protein.values())[0])
    
        for k in fasta_dict_site:
            fasta_dict_site[k] = fasta_dict_site[k] + ('_' * (length - len(fasta_dict_site[k])))
        fasta_site = '\n'.join(f'> {k}' + '\n' + fasta_dict_site.get(k, '_' * length) for k in fasta_dict_protein)
        with open(p, 'w') as f:
            f.write(fasta_site)
        return


class Mol2_File():

    def __init__(self, parent: Mol2, path: Path, type_: str, ischain: bool = False, chain_name: str = None):

        self._parent = parent
        self._path_object = path
        self._type = type_
        self._is_chain = ischain
        self._chain_name = chain_name

        self._atoms_df = None
        self._bonds_df = None
        self._residues_df = None
        self._substructures_df = None

        self._chains = []

        if not self._is_chain:
            chain_names = self.Chain_Names
            if chain_names:
                for name in chain_names:
                    chain_object = Mol2_File(self._parent, path, self._type, ischain=True, chain_name=name)

                    chain_object._atoms_df = self.Atoms.copy()
                    chain_object._atoms_df = chain_object._atoms_df.loc[chain_object._atoms_df['Chain'] == name]

                    chain_object._bonds_df = self.Bonds.copy()
                    chain_object._bonds_df = chain_object._bonds_df.loc[chain_object._bonds_df['Chain'] == name]

                    chain_object._substructures_df = self.Substructures.copy()
                    chain_object._substructures_df = chain_object._substructures_df.loc[chain_object._substructures_df['Chain'] == name]

                    chain_object._residues_df = self.Residues.copy()
                    chain_object._residues_df = chain_object._residues_df.loc[chain_object._residues_df['Chain'] == name]

                    self._chains.append(chain_object)
        

    def __str__(self):
        return f'<Mol2 {self._type} {self._parent._scpdb.scpdb}>' if not self.IsChain else f'<Mol2 {self._type} Chain {self._chain_name} {self._parent._scpdb.scpdb}>'
    
    def __repr__(self):
        return str(self)
    
    @property
    def IsChain(self):
        return self._is_chain
    
    @property
    def Name(self):
        if not self.IsChain:
            return self._parent._scpdb.pdb
        return self._chain_name

    @property
    def exists(self):
        return self._path_object.exists
    
    @property
    def path(self):
        return self._path_object.path

    @property
    def dict(self):

        self._prefix = self._type if not self.IsChain else f'{self._type} Chain'
        self._dict = {
            'Number Of Atoms': self.Number_Of_Atoms,
            'Number Of Bonds': self.Number_Of_Bonds,
            'Number Of Substructures': self.Number_Of_Substructures,
            'Number Of Residues': self.Number_Of_Residues,
            'X Min': self.Coordinate_X_Min,
            'X Max': self.Coordinate_X_Max,
            'Y Min': self.Coordinate_Y_Min,
            'Y Max': self.Coordinate_Y_Max,
            'Z Min': self.Coordinate_Z_Min,
            'Z Max': self.Coordinate_Z_Max,
            'Cubic Volume': self.Cube_Volume,
            'Molecular Weight': self.Molecular_Weight}
        if self._type == 'Protein':
            self._dict.update({'Number Of Binding Residues': self.Number_Of_Binding_Residues})
            self._dict.update({'Percent Of Binding Residues': self.Percent_Of_Binding_Residues})
            if not self.IsChain:
                self._dict.update({'Number Of Chains': self.Number_Of_Chains,
                                    'Number Of Different Chains': self.Number_Of_Different_Chains,
                                    'Chain Configuration': self.Chain_Configuration,
                                    'Dominant_Chain': self.Dominant_Chain.Name})
        self._dict.update({'Atom Count ' + k: v for k, v in self.Atom_Counts.items()})
        self._dict.update({'Atom Percent ' + k: v for k, v in self.Atom_Percents.items()})
        self._dict.update({'Atom Type Count ' + k: v for k, v in self.Atom_Type_Counts.items()})
        self._dict.update({'Atom Type Percent ' + k: v for k, v in self.Atom_Type_Percents.items()})
        self._dict.update({'' + k: v for k, v in self.Charge_Properties.items()})
        self._dict.update({'Structure Count ' + k: v for k, v in self.Substructure_Counts.items()})
        self._dict.update({'Structure Percent ' + k: v for k, v in self.Substructure_Percents.items()})
        self._dict.update({'Structure Type Count ' + k: v for k, v in self.Substructure_Type_Counts.items()})
        self._dict.update({'Structure Type Percent ' + k: v for k, v in self.Substructure_Type_Percents.items()})
        self._dict.update({'Residue Count ' + k: v for k, v in self.Residue_Counts.items()})
        self._dict.update({'Residue Percent ' + k: v for k, v in self.Residue_Percents.items()})
        self._dict = {self._prefix + ' ' + k: self._dict[k] if not isinstance(self._dict[k], float) else round(self._dict[k], 5) for k in self._dict}

        if self._type in ['Protein', 'Site'] and not self.IsChain:

            self._dict.update(self.Dominant_Chain.dict)
            
        return self._dict
    
    @property
    def raw(self):
        return self._path_object.content
    
    def Get_Rotation_Matrix(self, i: int):

        row = self.Residues.iloc[i]

        return self._parent._residue_rotation_matrix(row['X ALPHA'], row['Y ALPHA'], row['Z ALPHA'],
                                                     row['X N'], row['Y N'], row['Z N'],
                                                     row['X C'], row['Y C'], row['Z C'])
    
    # Atoms:

    @property
    def Atom_Block(self):
        return self._parent._get_block(self.raw, 'ATOM')
    
    @property
    def Atom_DF(self):
        return self._parent._block_to_df(self.Atom_Block)
    
    @property
    def Atoms(self):

        if self._atoms_df is not None:
            return self._atoms_df

        df = self._parent._get_atoms(self.Atom_DF)
        if df is None:
            return None

        df_subs = self.Substructures[['Substructure ID', 'Chain']]
        df = pd.merge(df, df_subs, on='Substructure ID', how='left')

        self._atoms_df = df
        return df
    
    @property
    def Atoms_No_H(self):

        df = self.Atoms.copy()
        if df is None:
            return None
        
        df = df.loc[(df['Element'] != 'H') & (df['Substructure Code'].str.slice(0, 3) != 'HOH')]
        return df
    
    @property
    def Atom_Counts(self):

        df = self.Atoms
        if df is None:
            return {}
        
        counter = Counter(df['Element'].to_list())
        for element in self._parent._scpdb.MW:
            counter.update({element: 0})
    
        return dict(counter)
    
    @property
    def Atom_Percents(self):

        counts = self.Atom_Counts

        return {k: round(100*v/self.Number_Of_Atoms, 5) if self.Number_Of_Atoms > 0 else 0 for k, v in counts.items()}
    
    
    @property
    def TRIPOS_Type_Counts(self):

        df = self.Atoms
        if df is None:
            return {}
        
        counter = Counter(df['TRIPOS'].to_list())
        return dict(counter)
    
    @property
    def Atom_Type_Counts(self):

        df = self.Atoms
        if df is None:
            return {}
        
        counter = Counter(df['Type'].to_list())
        return dict(counter)
    
    @property
    def Atom_Type_Percents(self):

        df = self.Atoms
        if df is None:
            return {}
        
        counts = self.Atom_Type_Counts
        return {k: round(100*v/self.Number_Of_Atoms, 5) if self.Number_Of_Atoms > 0 else 0 for k, v in counts.items()}
    
    @property
    def Total_Charge(self):

        df = self.Atoms
        if df is None:
            return None
    
        return df['Atom Charge'].sum()
    
    @property
    def Number_Of_Charged(self):

        df = self.Atoms
        if df is None:
            return None
    
        return sum(df['Atom Charge'] != 0)
    
    @property
    def Number_Of_PosCharged(self):

        df = self.Atoms
        if df is None:
            return None
    
        return sum(df['Atom Charge'] > 0)
    
    @property
    def Number_Of_NegCharged(self):

        df = self.Atoms
        if df is None:
            return None
    
        return sum(df['Atom Charge'] < 0)
    
    @property
    def Total_PosCharge(self):

        df = self.Atoms
        if df is None:
            return None
    
        return df.loc[df['Atom Charge'] > 0, 'Atom Charge'].sum()
    
    @property
    def Total_NegCharge(self):

        df = self.Atoms
        if df is None:
            return None
    
        return df.loc[df['Atom Charge'] < 0, 'Atom Charge'].sum()
    
    @property
    def Charge_Properties(self):
    
        return {f'{self._type} Total Charge': self.Total_Charge,
                f'{self._type} Total Negative Charge': self.Total_NegCharge,
                f'{self._type} Total Positive Charge': self.Total_PosCharge,
                f'{self._type} Number of Charged Atoms': self.Number_Of_Charged,
                f'{self._type} Number of Neg Charged Atoms': self.Number_Of_NegCharged,
                f'{self._type} Number of Pos Charged Atoms': self.Number_Of_PosCharged}
    
    @property
    def Number_Of_Atoms(self):

        df = self.Atoms
        if df is None:
            return 0
    
        return len(df)
    
    @property
    def Coordinate_X_Min(self):

        df = self.Atoms
        if df is None:
            return 0
    
        return df['X'].min()
    
    @property
    def Coordinate_X_Max(self):

        df = self.Atoms
        if df is None:
            return 0
    
        return df['X'].max()
    
    @property
    def Coordinate_Y_Min(self):

        df = self.Atoms
        if df is None:
            return 0
    
        return df['Y'].min()
    
    @property
    def Coordinate_Y_Max(self):

        df = self.Atoms
        if df is None:
            return 0
    
        return df['Y'].max()
    
    @property
    def Coordinate_Z_Min(self):

        df = self.Atoms
        if df is None:
            return 0
    
        return df['Z'].min()
    
    @property
    def Coordinate_Z_Max(self):

        df = self.Atoms
        if df is None:
            return 0
    
        return df['Z'].max()
    
    @property
    def Cube_Volume(self):

        df = self.Atoms
        if df is None:
            return 0
    
        return (df['X'].max() - df['X'].min()) * (df['Y'].max() - df['Y'].min()) * (df['Z'].max() - df['Z'].min())
    
    @property
    def Molecular_Weight(self):

        df = self.Atoms
        if df is None:
            return 0
    
        return df.loc[df['Substructure Name'] != 'HOH', 'Element'].map(self._parent._scpdb.MW).sum()
    
    # Bonds:

    @property
    def Bond_Block(self):
        return self._parent._get_block(self.raw, 'BOND')
    
    @property
    def Bond_DF(self):
        return self._parent._block_to_df(self.Bond_Block)
    
    @property
    def Bonds(self):
        if self._bonds_df is not None:
            return self._bonds_df
        
        df = self._parent._get_bonds(self.Bond_DF)
        atoms = self.Atoms.copy()
        if atoms is None:
            return None
        
        atoms = atoms[['Atom ID', 'Chain']].rename(columns={'Atom ID': 'Atom ID 1'})
        df = pd.merge(df, atoms, on='Atom ID 1', how='left')
        
        self._bonds_df = df
        return df
    
    @property
    def Bonds_No_H(self):

        df = self.Bonds.copy()
        if df is None:
            return None
        
        all_atoms = self.Atoms
        if all_atoms is not None:
            hydrogen_indices = all_atoms.loc[all_atoms['Element'] == 'H', 'Atom ID'].to_list()
            df = df.loc[~(df['Atom ID 1'].isin(hydrogen_indices) | df['Atom ID 2'].isin(hydrogen_indices))]
        return df
    
    @property
    def Bond_Counts(self):

        df = self.Bonds
        if df is None:
            return {}
        
        counter = Counter(df['Bond Type'].to_list())
    
        return dict(counter)
    
    @property
    def Number_Of_Bonds(self):

        df = self.Bonds
        if df is None:
            return 0
    
        return len(df)
    
    # Substructures:

    @property
    def Substructure_Block(self):
        return self._parent._get_block(self.raw, 'SUBSTRUCTURE')
    
    @property
    def Substructure_DF(self):
        return self._parent._block_to_df(self.Substructure_Block)
    
    @property
    def Substructures(self):
        if self._substructures_df is not None:
            return self._substructures_df
        
        df = self._parent._get_substructures(self.Substructure_DF)
        self._substructures_df = df
        return df
    
    @property
    def Substructures_No_Water(self):

        df = self.Substructures.copy()
        if df is None:
            return None
        
        df = df.loc[df['Substructure Name'] != 'HOH']

        return df
    
    @property
    def Substructure_Counts(self):

        df = self.Substructures
        if df is None:
            return {}
        
        counter = Counter(df['Substructure Name'].to_list())
    
        return dict(counter)
    
    @property
    def Substructure_Percents(self):

        counter = self.Substructure_Counts
    
        return {k: round(100 * v / self.Number_Of_Substructures, 5) if self.Number_Of_Substructures > 0 else 0 for k, v in counter.items()}
    
    @property
    def Substructure_Type_Counts(self):

        df = self.Substructures
        if df is None:
            return {}
        
        counter = Counter(df['Substructure Type'].to_list())
    
        return dict(counter)
    
    @property
    def Substructure_Type_Percents(self):

        counter = self.Substructure_Type_Counts
    
        return {k: round(100 * v / self.Number_Of_Substructures, 5) if self.Number_Of_Substructures > 0 else 0 for k, v in counter.items()}
    
    @property
    def Number_Of_Substructures(self):

        df = self.Substructures
        if df is None:
            return 0
    
        return len(df)
    
    @property
    def Chain_Names(self):

        if self.IsChain:
            return []

        df = self.Substructures
        if df is None:
            return None
    
        return df['Chain'].unique().tolist()
    
    @property
    def Chains(self):
        return self._chains
    
    @property
    def Number_Of_Chains(self):
        if self.IsChain:
            return 0
        return len(self.Chains)
    
    @property
    def Number_Of_Different_Chains(self):
        if self.IsChain:
            return 0
        return len(set(self.FASTA.values()))
    
    @property
    def Chain_Configuration(self):
        if self.IsChain:
            return None
        fasta = self.FASTA
        
        counter = Counter(list(fasta.values()))
        config = sorted(list(counter.values()), reverse=True)

        return '+'.join(str(i) for i in config)

    def Get_Chain(self, name: str):
        if self.IsChain:
            return None
        for chain in self.Chains:
            if chain.Name.lower() == name.lower():
                return chain
        return None
    
    # Residues

    @property
    def Residues(self):

        if self._residues_df is not None:
            return self._residues_df

        df = self.Substructures.copy()
        if df is None:
            return None
        
        df = df.loc[df['Substructure Type'] == 'AA']
        df['Substructure Letter'] = df['Substructure Name'].map(self._parent._scpdb.ONE_LETTER)
        df['Residue Group'] = df['Substructure Name'].map(self._parent._scpdb.RESIDUES_GROUPS)

        atoms = self.Atoms_No_H
        if atoms is None:
            return df
        
        atoms = atoms.loc[atoms['Substructure Type'] == 'AA']

        alpha_carbons = atoms.copy()
        alpha_carbons = alpha_carbons.loc[alpha_carbons['TRIPOS'] == 'CA', ['Substructure ID', 'X', 'Y', 'Z']]
        alpha_carbons = alpha_carbons.rename(columns={'X': 'X ALPHA', 'Y': 'Y ALPHA', 'Z': 'Z ALPHA'})
        peptide_nitrogens = atoms.copy()
        peptide_nitrogens = peptide_nitrogens.loc[peptide_nitrogens['TRIPOS'] == 'N', ['Substructure ID', 'X', 'Y', 'Z']]
        peptide_nitrogens = peptide_nitrogens.rename(columns={'X': 'X N', 'Y': 'Y N', 'Z': 'Z N'})
        peptide_carbons = atoms.copy()
        peptide_carbons = peptide_carbons.loc[peptide_carbons['TRIPOS'] == 'C', ['Substructure ID', 'X', 'Y', 'Z']]
        peptide_carbons = peptide_carbons.rename(columns={'X': 'X C', 'Y': 'Y C', 'Z': 'Z C'})

        atoms['Mass'] = atoms['Element'].map(self._parent._scpdb.MW)
        atoms['XMass'] = atoms['X'].astype('float') * atoms['Mass']
        atoms['YMass'] = atoms['Y'].astype('float') * atoms['Mass']
        atoms['ZMass'] = atoms['Z'].astype('float') * atoms['Mass']

        atoms = atoms[['Substructure ID', 'Mass', 'XMass', 'YMass', 'ZMass']].groupby('Substructure ID', as_index=False).sum()
        atoms['X MASS'] = atoms['XMass'] / atoms['Mass']
        atoms['Y MASS'] = atoms['YMass'] / atoms['Mass']
        atoms['Z MASS'] = atoms['ZMass'] / atoms['Mass']

        atoms = atoms[['Substructure ID', 'X MASS', 'Y MASS', 'Z MASS']]
        df = pd.merge(df, atoms[['Substructure ID', 'X MASS', 'Y MASS', 'Z MASS']], how='left', on='Substructure ID')
        df = pd.merge(df, alpha_carbons[['Substructure ID', 'X ALPHA', 'Y ALPHA', 'Z ALPHA']], how='left', on='Substructure ID')
        df = pd.merge(df, peptide_nitrogens[['Substructure ID', 'X N', 'Y N', 'Z N']], how='left', on='Substructure ID')
        df = pd.merge(df, peptide_carbons[['Substructure ID', 'X C', 'Y C', 'Z C']], how='left', on='Substructure ID')

        if self._type == 'Ligand':
            df['Binding'] = 0
        elif self._type == 'Site':
            df['Binding'] = 1
        else:
            binding_residues = self._parent.site.Residues
            df = pd.merge(df, binding_residues[['Substructure Code', 'Chain', 'Binding']], how='left', on=['Substructure Code', 'Chain'])
            df['Binding'] = df['Binding'].fillna(0)

        df['Binding'] = df['Binding'].astype(int)
        self._residues_df = df
        return df
    

    def Save_Residues(self):
        p = os.path.join(self._parent._scpdb.folder_path, 'residues.csv')
        self.Residues.to_csv(p)
        return
    
    def Save_Atoms(self):
        p = os.path.join(self._parent._scpdb.folder_path, 'atoms.csv')
        self.Atoms_No_H.to_csv(p)
        return

    @property
    def Residue_Counts(self):

        df = self.Residues
        if df is None:
            return {}
        
        counter = Counter(df['Substructure Letter'].to_list())
        for res in self._parent._scpdb.ONE_LETTER.values():
            counter.update({res: 0})

        return dict(counter)
    
    @property
    def Residue_Percents(self):

        counts = self.Residue_Counts

        return {k: round(100*v/self.Number_Of_Residues, 5) if self.Number_Of_Residues > 0 else 0 for k, v in counts.items()}
    
    @property
    def Number_Of_Residues(self):

        df = self.Residues
        if df is None:
            return 0

        return len(df)

    @property
    def FASTA(self):

        df = self.Residues
        if df is None:
            return None

        if self.IsChain:

            if len(df) == 0:
                return ''
            residues, positions = df['Substructure Letter'], df['Substructure Position']
            position_dict = {pos: res for pos, res in zip(positions, residues)}
            for pos in range(1, positions.max()+1):
                if pos not in position_dict:
                    position_dict.update({pos: '_'})

            return ''.join(position_dict[i] for i in range(1, positions.max()+1))

        max_length_fasta = -1
        fasta = {}

        for chain in self.Chains:
            f = chain.FASTA
            if len(f) > max_length_fasta:
                max_length_fasta = len(f)
            fasta.update({chain.Name:f})
        
        for chain in fasta:
            fasta[chain] = fasta[chain] + ('_' * (max_length_fasta - len(fasta[chain])))
        return fasta
    
    @property
    def Number_Of_Binding_Residues(self):
        
        df = self.Residues
        if df is None:
            return 0
        
        return df['Binding'].sum()
    
    @property
    def Percent_Of_Binding_Residues(self):
        
        df = self.Residues
        if df is None:
            return 0
        
        return (100 * df['Binding'].sum() / len(df)) if len(df) > 0 else 0.0
    
    @property
    def Dominant_Chain(self):

        dominant = None
        max_score = -1

        for chain in self.Chains:

            score = chain.Number_Of_Binding_Residues
            if score > max_score:
                max_score = score
                dominant = chain
        return dominant
