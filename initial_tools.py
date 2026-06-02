import requests
import os
import time
import random
import importlib
import sys
import json
import warnings

from typing import Collection, Any

import pandas as pd
import numpy as np
import regex as re

from tqdm import tqdm
from bs4 import BeautifulSoup

from pathlib import Path as PathLib

pd.options.display.max_colwidth = 1000
pd.options.display.max_columns = 20
warnings.simplefilter("ignore")



def print_dependencies():
    """
    Helper function to print the used dependencies
    """

    modules = [
        "requests", "pandas", "numpy", "regex", "tqdm",
        "bs4", "matplotlib", "torch", "sklearn", "scipy", 
        "seaborn", "torch_geometric"
    ]

    print(" Dependencies ".center(40, '-'))
    print(f"Python version: {sys.version.split()[0]}")
    print("-" * 40)

    for m in modules:
        try:
            mod = importlib.import_module(m)
            version = getattr(mod, "__version__", None)
            if version is None and m == "bs4":
                import bs4
                version = getattr(bs4, "__version__", "unknown")
            print(f"{m:<16}: {version if version else 'builtin / unknown'}")
        except Exception as e:
            print(f"{m:<16}: not installed ({e.__class__.__name__})")

    print("-" * 40)

def format_size(size: float | int):
        if size > 1024 * 1024 * 1024:
            size /= 1024 * 1024 * 1024
            return f'{round(size, 1)} Gb'
        elif size > 1024 * 1024:
            size /= 1024 * 1024
            return f'{round(size, 1)} Mb'
        elif size > 1024:
            size /= 1024
            return f'{round(size, 0)} Kb'
        else:
            return f'{round(size, 0)} b'

def checkin(argument: Any,
            name: str,
            allowed_types = None,
            allowed_span: tuple | list = None,
            allowed_values: Collection = None):
    """
    Main method for assertion

    Args:
        argument (Any): variable to check
        name (str): variable name
        allowed_types (_type_, optional): datatypes allowed. Defaults to None.
        allowed_span (tuple | list, optional): spans allowed. Defaults to None.
            If span is a tuple (a, b): strict inequalities assumed;
            If span is a list [a, b]: non-strict inequalities assumed;
            Span limits can be np.inf or -np.inf.
        allowed_values (Collection, optional): values allowed. Defaults to None.
    """


    if allowed_types is not None:
        if isinstance(argument, allowed_types):
            pass
        else:
            raise TypeError(f'{name} must be of types: {allowed_types}; not {type(argument)} ({argument}')

    if allowed_span is not None:
        if isinstance(allowed_span, tuple):
            strict = True
        elif isinstance(allowed_span, list):
            strict = False
        else:
            raise TypeError(f'allowed_span must be tuple | list, not {type(allowed_span)}')

        if strict:
            if argument <= allowed_span[0] and allowed_span[0] > -np.inf:
                raise ValueError(f'{name} must be > {allowed_span[0]}, not {argument}')
            if argument >= allowed_span[1] and allowed_span[1] < np.inf:
                raise ValueError(f'{name} must be < {allowed_span[1]}, not {argument}')
        else:
            if argument < allowed_span[0] and allowed_span[0] > -np.inf:
                raise ValueError(f'{name} must be >= {allowed_span[0]}, not {argument}')
            if argument > allowed_span[1] and allowed_span[1] < np.inf:
                raise ValueError(f'{name} must be <= {allowed_span[1]}, not {argument}')

    if allowed_values is not None:

        if argument not in allowed_values:
            raise KeyError(f'{name} must be one of: [{", ".join(allowed_values)}], not {argument}')

def structure_printer(structure: dict, _prefix=""):
    """
    A tool to print down the file structure from its dict form

    Args:
        structure (dict): dictionary structure
        _prefix (str, optional): auxillary variable for rrecursion. Defaults to "".
    """

    checkin(structure, 'structure', dict)

    items = list(structure.items())

    for i, (name, value) in enumerate(items):
        connector = "└── " if i == len(items) - 1 else "├── "
        print(_prefix + connector + name)

        if isinstance(value, dict):
            extension = "    " if i == len(items) - 1 else "│   "
            structure_printer(value, _prefix + extension)

def check_file_structure(base_path: PathLib, expected_structure: dict):
    """
    Checks if the file structure corresponds to the expected structure
    starting from the root folder (base_path)

    Note:
    - missing / extra repositories are detected
    - missing files are detected
    - extra files are NOT detected

    Args:
        base_path (Path): the root repository
        expected_structure (dict): dictionary of expected structure

    Returns:
        dict: missing directories, extra directories and missing files

    """

    checkin(expected_structure, 'structure', dict)
    checkin(base_path, 'base_path', PathLib)

    missing_files = []
    missing_dirs = []
    extra_dirs = []

    def _check(current_path: PathLib, expected_dict: dict):
        """
        Helper recursive tool
        """

        # Catch missing directory
        if not current_path.exists():
            missing_dirs.append(str(current_path))
            return

        # Extract actual and expected nested directories
        actual_dirs = {p.name for p in current_path.iterdir() if p.is_dir()}
        expected_dirs = {k for k, v in expected_dict.items() if isinstance(v, dict)}

        # Detect extra directories
        for d in actual_dirs - expected_dirs:

            if d != "__pycache__":
                extra_dirs.append(str(current_path / d))

        # Check expected elements
        for name, value in expected_dict.items():
            path = current_path / name

            # Detect missing repository
            if isinstance(value, dict):
                if not path.exists():
                    missing_dirs.append(str(path))
                else:
                    _check(path, value)

            # Detect missing files
            else:
                if not path.exists():
                    missing_files.append(str(path))

    _check(base_path, expected_structure)

    print('File Structure Check Report:\n' + '-'*26 + '\nThe expected structure:\n')
    structure_printer(expected_structure)
    print('\nMissing directories:\n\t' + ('\n\t'.join(missing_dirs) if missing_dirs else 'None'))
    # print('\nExtra directories:\n\t' + ('\n\t'.join(extra_dirs) if extra_dirs else 'None'))
    print('\nMissing files:\n\t' + ('\n\t'.join(missing_files) if missing_files else 'None'))

    return {
        "missing_files": missing_files,
        "missing_dirs": missing_dirs,
        "extra_dirs": extra_dirs,
    }

def download_scPDB(list_of_entries: Collection[str] = None,
                   files_to_load: Collection[str] = {'protein.mol2', 'site.mol2'},
                   skip_existing_files: bool = True,
                   version: str = 'all',
                   dir: str = 'Files/scPDB'):
    """
    Allows to download scPDB respository (completely or partially).
    If no entries provided, it s expected to find a .txt file with each line corresponding to an entry
    in the current directory.

    IMPORTANT:
    scPDB entries are like PDB IDs, but also have upload index / version, e.g. "20gs" -> "20gs_2"
    If only PDB IDs are provided, the code will try to find its uploads in scPDB, which might take more time.
    If version='all': all found uploads will be downloaded. (If version='latest': only the latest upload)

    In the directory (dir) a separate folder for each entry will be cerated. Folder name will be its scPDB name (with version)
    In each folder files corresponding to this entry will be stored.

    If skip_existing_files is True, only missing files will be downloaded.

    ATTENTION:
    The total size of scPDB database is > 15Gb. The major part is from protein.mol2 files.

    Files that failed to be downloaded will be stores in a 'failed_downloads.csv' in the current directory

    Args:
        list_of_entries (Collection[str], optional): list of scPDB entries or PDB IDs to download. Defaults to None.
        files_to_load (Collection[str], optional): types of files to download,
            possible types: 'protein.mol2', 'site.mol2', 'cavity.mol2', 'IPF.txt', 'interaction.mol2', 'ligand.mol2'. Defaults to {'protein.mol2', 'site.mol2'}.
        skip_existing_files (bool, optional): skip files that are already in the directory. Defaults to True.
        version (str, optional): what uploads to download from scPDB (only important when PDB IDs provided).
            Possible options: 'all', 'latest'. Defaults to 'all'.
        dir (str, optional): directory where to store the database. Defaults to 'Files/scPDB'.

    """
    
    # Check that the list of entries is of correct dtype
    if list_of_entries is not None:
        assert isinstance(list_of_entries, Collection), f'list_of_entries must be a Collection, not {type(list_of_entries)}'
        for entry in list_of_entries:
            assert isinstance(entry, str), f'each entry in list_of_entries must be a str, not {type(entry)}'
    else:
        with open('all_scpdb_entries.txt', 'r') as file:
            list_of_entries = file.read().split('\n')
            list_of_entries = list(filter(lambda x: x, list_of_entries))
    list_of_entries = set(list_of_entries)

    # Check if files to load are of correct dtype and exist
    all_possible_files = {'protein.mol2', 'site.mol2', 'cavity.mol2', 'IPF.txt', 'interaction.mol2', 'ligand.mol2'}
    expected_kbytes = {'protein.mol2': 900_000, 'site.mol2': 60_000, 'cavity.mol2': 6_000, 'IPF.txt': 1_000, 'interaction.mol2': 6_000, 'ligand.mol2': 6_000}
    expected_size = 0
    if files_to_load is not None:
        assert isinstance(files_to_load, Collection), f'files_to_load must be a Collection, not {type(files_to_load)}'
        files_to_load = set(files_to_load)
        for file in files_to_load:
            assert isinstance(file, str), f'each file in files_to_load must be a str, not {type(entry)}'
            assert file in all_possible_files, f'unknown file: {file}, consider: {all_possible_files}'
            expected_size += expected_kbytes[file]


    # Check the dtypes of other keywords
    assert isinstance(skip_existing_files, bool), f'skip_existing_files must be bool, not {type(skip_existing_files)}'
    assert isinstance(version, str), f'version must be a str, not {type(version)}'
    assert version in {'latest', 'all'}, f'version can only be "latest" or "all", not {version}'
    
    
    # Estimate the total size of the downloads
    expected_size *= len(list_of_entries)
    
    # Report
    print('Downloading scPDB files')
    print(f'Entries provided: {len(list_of_entries)}')
    print(f'Files: {files_to_load}')
    print(f'Expected size of the downloads: {format_size(expected_size)}')
        

    # Downloading
    base_path = 'http://bioinfo-pharma.u-strasbg.fr/scPDB/EXPORTENTRY='
    # http://bioinfo-pharma.u-strasbg.fr/scPDB/EXPORTENTRY=PROT=1hso_1
    file_suffix = {'protein.mol2': 'PROT=',
                    'site.mol2': 'SITE=',
                    'cavity.mol2': 'CAVITY=',
                    'IPF.txt': 'IFP=',
                    'interaction.mol2': 'INTS=',
                    'ligand.mol2': 'LIG='}
    true_size = 0
    failed_entries = []
    skipped_existing = 0

    
    with tqdm(total=len(list_of_entries) * len(files_to_load), desc='Downloading...') as pbar:

        # Iterate through provided entries
        for _i, entry in enumerate(sorted(list_of_entries)):

            pbar.set_description(f'Downloading entry {_i} ({entry}) ({format_size(true_size)})...')

            # If entry number is explicitly provided: just use it...
            if '_' in entry:
                valid_versions = [entry]

            # ...otherwise try different versions
            else:
                valid_versions = []
                for v in range(1, 21):

                    path = base_path + file_suffix[sorted(files_to_load)[0]] + entry + '_' + str(v)
                    response = requests.get(path)
                    if response.status_code != 200 or not ('TRIPOS' in response.text):
                        continue
                    valid_versions.append(entry + '_' + str(v))

            # If no valid versions found - skip the entry
            if not valid_versions:

                for f_ in files_to_load:
                    failed_entries.append([entry, f_])
                pbar.update(len(files_to_load))
                continue

            # Only take the lastest added entry if required
            if version == 'latest':
                valid_versions = [valid_versions[-1]]

            # Iterate through found URLs
            for entry_v in valid_versions:

                # Create a new folder (optionally)
                os.makedirs(os.path.join(dir, entry_v), exist_ok=True)

                # Iterate through files to load
                for file in files_to_load:

                    # Skip the step if allowed
                    if skip_existing_files:
                        if os.path.isfile(os.path.join(dir, entry_v, file)):

                            with open(os.path.join(dir, entry_v, file), 'r') as f:
                                ff = f.read()

                            if '.mol2' in file:
                                if 'ATOM' in ff and 'BOND' in ff and 'SUBSTRUCTURE' in ff:
                                    pbar.update(1)
                                    skipped_existing += 1
                                    continue
                            else:
                                if not 'html' in ff:
                                    pbar.update(1)
                                    skipped_existing += 1
                                    continue
                    
                    url = base_path + file_suffix[file] + entry_v
                    response = requests.get(url, cert=False, )
                    if response.status_code != 200 or 'html' in response.text:
                        failed_entries.append([entry_v, file])
                        pbar.update(1)
                        continue
                    else:
                        text = response.text
                        with open(os.path.join(dir, entry_v, file), 'w') as f:
                            f.write(text)
                        true_size += os.path.getsize(os.path.join(dir, entry_v, file))

                    pbar.update(1)

    failed_entries = pd.DataFrame(failed_entries, columns=['entry', 'file'])
    os.makedirs('Logs', exist_ok=True)
    failed_entries.to_csv('Logs/scpdb_failed_downloads.csv', sep=',')
    print(f'Downloading finished (actual size: {format_size(true_size)}; skipped as existing: {skipped_existing})')
    print(f'Failed entries: {len(failed_entries.entry.unique())} (files: {len(failed_entries)})')

    return {f: os.path.join(dir, f) for f in os.listdir(dir)}

def download_SCOPe(version: str = '2.08',
                   exist_ok: bool = True,
                   dir: str = 'Files/SCOPe'):
    """
    Allows to dowload SCOPe classification of PDB entries (and their chains),
    the version of SCOPe is selectable, directory to store is selectable

    Args:
        version (str, optional): version of SCOPe to download. Recommended 2.08 or 2.07. Defaults to '2.08'.
        exist_ok (bool, optional): allows to skip if CSV is already in the directory. Defaults to True.
        dir (str, optional): directory to store the file to. Defaults to 'Files/SCOPe'.
    """

    # Check the validity of the input
    allowed_versions = {'2.08', '2.07', '2.06', '2.05', '2.04', '2.03', '2.02', '2.01',
                        '1.75', '1.73', '1.71', '1.69', '1.67', '1.65', '1.63', '1.61',
                        '1.59', '1.57', '1.55'}
    assert isinstance(version, str), f'version must be a str, not {type(version)}'
    assert version in allowed_versions, f'uknown version "{version}", consider: {allowed_versions}'
    assert isinstance(exist_ok, bool), f'exist_ok must be bool, not {type(exist_ok)}'
    assert isinstance(dir, str), f'dir must be str, not {type(dir)}'

    # Report
    print('Downloading SCOPe')
    print(f'Version: {version}')
    path = os.path.join(dir, version.replace('.', '_') + '.csv')
    print(f'Path: {path}')

    # Ensure that dir exists and skip if file is already there
    os.makedirs(dir, exist_ok=True)
    if os.path.isfile(path):
        print(f'Version is already in the directory ({format_size(os.path.getsize(path))})')
        return

    # Downloading
    base_url = 'https://scop.berkeley.edu/downloads/parse/dir.cla.scope.@version@-stable.txt'
    url = base_url.replace('@version@', version)
    response = requests.get(url, verify=False)
    if response.status_code != 200:
        print(f'Unfortunately URL request was not successful, code = {response.status_code}')
        return

    # Process the data and store it as a csv file (through pandas)
    csv = response.text.split('\n')
    csv = list(filter(lambda x: x and x[0] != '#', csv))
    csv = [line.split('\t')[:4] for line in csv]
    csv = pd.DataFrame(csv, columns=['scope_id', 'pdb_id', 'chain', 'class'])
    csv['chain'] = csv['chain'].str.split(':').apply(lambda x: x[0])
    csv.to_csv(path)

    # Report
    print(f'Version saved to the directory ({format_size(os.path.getsize(path))})')

    return

def download_SIFTS(exist_ok: bool = True,
                   dir: str = 'SIFTS'):
    """
    Allows to dowload SIFT files of PDB entries (and their chains)

    Args:
        exist_ok (bool, optional): allows to skip if CSV is already in the directory. Defaults to True.
        dir (str, optional): directory to store the file to. Defaults to 'SIFTS'.
    """

    # Check the validity of the input
    assert isinstance(exist_ok, bool), f'exist_ok must be bool, not {type(exist_ok)}'
    assert isinstance(dir, str), f'dir must be str, not {type(dir)}'

    # Report
    print('Downloading SIFTS')
    print(f'Path: {dir}')

    # Files
    files = {
        'pdb_chain_enzyme.csv': 'https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_enzyme.csv.gz',
        'pdb_chain_uniprot.csv': 'https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz',
        'scop.csv': 'https://www.ebi.ac.uk/pdbe/scop/files/scop-cla-latest.txt',
        'scop_names.csv': 'https://www.ebi.ac.uk/pdbe/scop/files/scop-des-latest.txt'
    }

    # Ensure that dir exists and skip if file is already there
    os.makedirs(dir, exist_ok=True)
    existing_files = os.listdir(dir)

    total_size = 0

    # Downloading
    for file, url in files.items():

        # Skip if exists
        if file in existing_files and exist_ok:
            print(f'{file} already exists - skipped')
            continue

        save_path = os.path.join(dir, file)

        try:
        
            if file == 'pdb_chain_enzyme.csv':

                df_ec = pd.read_csv(url, compression="gzip", skiprows=1)
                df_ec.to_csv(save_path, index=False)

            elif file == 'pdb_chain_uniprot.csv':

                df_uniprot = pd.read_csv(url, compression="gzip", skiprows=1)
                df_uniprot.to_csv(save_path, index=False)

            elif file == 'scop.csv':
                
                df_scop = requests.get(url).content.decode('utf-8')
                df_scop = [line.removeprefix('# ').split(' ') for line in df_scop.split('\n') if not line.startswith('#') or 'FA-' in line]
                df_scop = pd.DataFrame(df_scop[1:], columns=df_scop[0])
                df_scop.to_csv(save_path, index=False)
            
            elif file == 'scop_names.csv':
                
                df_scop_names = requests.get(url).content.decode('utf-8')
                df_scop_names = [line.replace(',', '/') for line in df_scop_names.split('\n') if not line.startswith('#')]
                df_scop_names = [(line.split(' ')[0], ' '.join(line.split(' ')[1:])) for line in df_scop_names]
                df_scop_names = pd.DataFrame(df_scop_names, columns=['Code', 'Description'])
                df_scop_names.to_csv(save_path, index=False)
            
            total_size += os.path.getsize(save_path)
            print(f'{file} downloaded successfully ({format_size(os.path.getsize(save_path))})')

        except Exception as e:
            print(f'Error downloading {file}: {e}')

    # Report
    print(f'Files saved to the directory ({format_size(total_size)})')

    return

def download_scPDB_web_pages(exist_ok: bool = True,
                             dir: str = 'Files/scPDB',
                             exclude_entries: Collection = None):
    """
    Allows to download source codes of scPDB web-pages for the
    entries stored in scpdb_dir (by default: 'Files/scPDB')

    The source codes are then stored in the same dir as html.txt files.

    IMPORTANT: the procedure might take 6-12 h depending on the Internet connection.

    Args:
        exist_ok (bool, optional): allows to skip already saved pages. Defaults to True.
        dir (str, optional): directory where to store page sources. Defaults to 'Files/scPDB'.
        exclude_entries (Collection, optional): exclude some entries. Defaults to None.
    """

    def get_source_code(entry_id):
        """
        Helper function to fetch the source code
        """

        url = f"http://bioinfo-pharma.u-strasbg.fr/scPDB/SITE={entry_id}"
        r = requests.get(url)
        if r.status_code != 200:
            return r.status_code
        return r.text

    # Check the input
    assert isinstance(exclude_entries, Collection), f'exclude_entries must be a str, not {type(exclude_entries)}'
    assert isinstance(exist_ok, bool), f'exist_ok must be bool, not {type(exist_ok)}'
    assert isinstance(dir, str), f'dir must be str, not {type(dir)}'
    os.makedirs(dir, exist_ok=True)

    all_entries = sorted(set([file.replace('\n', '') for file in os.listdir(dir)]) - set(exclude_entries))

    # Report
    print('Downloading scPDB web pages')
    print(f'Entries provided (from {dir}): {len(all_entries)}')
    print(f'Expected size of the downloads: {format_size(40_000 * len(all_entries))}')

    true_size = 0
    skipped_existing = 0
    failed_entries = []

    # Iterate through entries
    for entry in tqdm(all_entries, desc='Loading scPDB web pages...', total=len(all_entries)):

        # Create the path and check if it already exixts
        path = os.path.join(dir, entry, f'html.txt')
        if exist_ok and os.path.isfile(path):
            skipped_existing += 1
            continue
        
        # Wait for a bit to avoid request errors
        lag = random.random() / 5
        time.sleep(lag)

        # Fetch the source code and save it
        page = get_source_code(entry)
        if not isinstance(page, int) and len(page) < 500_000:
            with open(path, 'w') as f:
                f.write(page)
            true_size += os.path.getsize(path)
        else:
            failed_entries.append([entry])

    failed_entries = pd.DataFrame(failed_entries, columns=['entry'])
    failed_entries.to_csv('Logs/failed_web_pages.csv', sep=',')
    print(f'Downloading finished (actual size: {format_size(true_size)}; skipped as existing: {skipped_existing})')
    print(f'Failed entries: {len(failed_entries)}')

def parse_table(table):
    """
    Helper function to parse html table (through bs4 utility)
    """
    t = []
    for row in table.find_all("tr"):
        cells = [c.get_text(strip=True) for c in row.find_all(["td","th"])]
        t.append(cells)

    return t

def parse_header(parser) -> dict:
    """
    Helper function

    Extracts:
    - PDB ID
    - PDB URL
    - resolution
    - method
    - deposition date

    using bs4 utility.

    Returns:
        dict
    """

    # Main information is stored in the From_Line divs.
    form_lines = parser.find_all("div", class_="Form_Line")

    # I expect 2 such divs. If not - the page is not valid
    if len(form_lines) < 2:
        return None

    # In the first Form_Line there should be PDB ID and Resolution
    first_form = form_lines[0].find_all("div", class_="InpT")
    pdb_a = first_form[0].find("a")
    pdb_id = pdb_a.get_text(strip=True) if pdb_a else None
    pdb_link = pdb_a["href"] if pdb_a else None
    resolution = first_form[1].get_text(strip=True)
    resolution = re.sub(r"[^0-9.\-]", "", resolution)

    # In the second Form_Line there should be method and deposition date
    second_form = form_lines[1].find_all("div", class_="InpT")
    method = second_form[0].get_text(strip=True).replace("Experimental Method :", "").strip()
    date = second_form[1].get_text(strip=True).replace("Deposition Date :", "").strip()

    return {
        "PDB ID": pdb_id,
        "PDB URL": pdb_link,
        "Resolution": resolution,
        "Method": method,
        "Deposition Date": date
    }

def parse_scpdb_source_code(entry: str,
                            source_code: str,
                            scope_df: pd.DataFrame) -> dict:
    """
    Allows to extract information from scPDB source code for an entry.

    Args:
        entry (str): scPDB entry
        source_code (str): source code as a str (already loaded)
        scope_df (pd.DataFrame): dataframe of SCOPe classes

    Returns:
        dict (dictionary with parsed information)
    """

    # Use bs4 to create an html parser
    parser = BeautifulSoup(source_code, "html.parser")
    entry = entry.removesuffix('.txt')
    base_path = 'http://bioinfo-pharma.u-strasbg.fr/scPDB'

    # Get main information
    info = {'scPDB ID': entry}
    info.update(parse_header(parser))

    # Protein Section
    protein_div = parser.find("div", {"class": "BOX1", "id": "ProtAnnot"})
    protein_body = protein_div.find("div", {"class": "BOX1-body"})

    # There should be 2 sections with tables (main and composition)
    uniprot_div = protein_body.find("div", {"id": "unip"})
    protein_tables = uniprot_div.find_all("table")

    composition_div = protein_body.find("div", {"id": "bsitecompos"})
    protein_composition_tables = composition_div.find_all("table")

    # Ligand Section
    ligand_div = parser.find("div", {"class": "BOX1", "id": "LigAnnot"})
    ligand_body = ligand_div.find("div", {"class": "BOX1-body"})

    # There should be 2 sections with tables (main and position)
    ligand_properties = ligand_body.find("div", {"id": "ligProps"})
    ligand_tables = ligand_properties.find_all("table")

    ligand_position = ligand_body.find("div", {"id": "ligPos"})
    ligand_position_tables = ligand_position.find_all("table")

    # The expected tables are:
    #   for protein:
    uniprot_table = parse_table(protein_tables[0])
    chains_table = parse_table(protein_tables[1])
    site_composition_table = parse_table(protein_composition_tables[0])
    cavity_table_1 = parse_table(protein_composition_tables[1])
    cavity_table_2 = parse_table(protein_composition_tables[2])
    #   for ligand:
    main_ligand_table = parse_table(ligand_tables[0])
    ligand_atoms_table = parse_table(ligand_tables[1])
    ligand_mass_center_table = parse_table(ligand_position_tables[0])

    # Extracting info from tables
    #   from uniprot table: Name, TaxID, Organism, Reign, AC, ID, EC
    for row in uniprot_table:
        if len(row) != 2:
            continue
        category, value = row
        if 'name' in category.lower():
            info.update({'Name': value})
        elif 'taxid' in category.lower():
            info.update({'TaxID': value})
        elif 'organism' in category.lower():
            info.update({'Organism': value})
        elif 'reign' in category.lower():
            info.update({'Reign': value})
        elif 'ac' in category.lower():
            info.update({'AC': value})
        elif 'id' in category.lower():
            info.update({'Uniprot ID': value})
        elif 'ec' in category.lower():
            info.update({'EC': value})

    #   from chain table: Percentage of Residues within binding site per Chain
    chains_with_percentages = []
    chains = []
    for row in chains_table[1:]:
        if len(row) != 2:
            continue
        chain, percent = row
        percent = percent.replace(' ', '').replace('%', '')
        chains_with_percentages.append(f'{chain} = {percent}')
        chains.append(chain)
    chains_with_percentages = ' / '.join(chains_with_percentages)
    info.update({'Percentage of Residues within binding site per Chain': chains_with_percentages})
    info.update({'Chains': ','.join(chains)})
    info.update({'Number of Chains': len(chains)})

    #   from site composition table: Metals, Water, Residues, B-Factor
    for row in site_composition_table:
        if len(row) != 2:
            continue
        category, value = row
        if 'metals' in category.lower():
            info.update({'Metals in Binding Site': value})
        elif 'cofactors' in category.lower():
            info.update({'Cofactors in Binding Site': value})
        elif 'water' in category.lower():
            info.update({'Water Molecules in Binding Site': value})
        elif 'non standard' in category.lower():
            info.update({'Non Standard Amino Acids in Binding Site': value})
        elif 'standard' in category.lower():
            info.update({'Standard Amino Acids in Binding Site': value})
        elif 'residues' in category.lower():
            info.update({'Number of Residues in Binding Site': value})
        elif 'factor' in category.lower():
            info.update({'B-Factor of Binding Site': value})

    #   from cavity tables: Ligandability, Volume, Hydrophobic Residues %, Polar Residues %
    info.update({'Cavity Ligandability': cavity_table_1[1][0]})
    info.update({'Cavity Volume (A^3)': cavity_table_1[1][1]})
    info.update({'Cavity % Hydrophobic': cavity_table_2[1][0]})
    info.update({'Cavity % Polar': cavity_table_2[1][1]})

    #   from ligand table: Surface Areas, DrugBank ID, Weight, Forula, HET
    for row in main_ligand_table:
        if len(row) != 2:
            continue
        category, value = row
        if 'polar' in category.lower():
            info.update({'Ligand Polar Surface Area (A^2)': re.sub(r"[^0-9.\-]", "", value)})
        elif 'buried' in category.lower():
            info.update({'Ligand Buried Surface Area (A^2)': re.sub(r"[^0-9.\-]", "", value)})
        elif 'drugbank' in category.lower():
            info.update({'Ligand DrugBank ID': value})
        elif 'weight' in category.lower():
            info.update({'Ligand Molecular Weight': re.sub(r"[^0-9.\-]", "", value)})
        elif 'formula' in category.lower():
            info.update({'Ligand Formula': value})
        elif 'het' in category.lower():
            info.update({'Ligand Het Code': value})
    
    #   from liagnd atoms table: Donors, Acceptors, Rings, Charges atoms, Rotatable bonds
    for row in ligand_atoms_table:
        if len(row) != 2:
            continue
        category, value = row
        if 'acceptor' in category.lower():
            info.update({'Ligand H-Bond Acceptors': value})
        elif 'donor' in category.lower():
            info.update({'Ligand H-Bond Donors': value})
        elif 'aromatic' in category.lower():
            info.update({'Ligand Aromatic Rings': value})
        elif 'rings' in category.lower():
            info.update({'Ligand Rings': value})
        elif 'anion' in category.lower():
            info.update({'Ligand Anionic Atoms': value})
        elif 'cation' in category.lower():
            info.update({'Ligand Cationic Atoms': value})
        elif 'five' in category.lower():
            info.update({'Ligand Rule of Five Violation': value})
        elif 'rotatable' in category.lower():
            info.update({'Ligand Rotatable Bonds': value})

    #   from liagnd mass center table: center of masses of a ligand
    info.update({'Ligand Mass Center (X)': ligand_mass_center_table[1][0]})
    info.update({'Ligand Mass Center (Y)': ligand_mass_center_table[1][1]})
    info.update({'Ligand Mass Center (Z)': ligand_mass_center_table[1][2]})

    # Now we also add links to dowloadable files
    info.update({'scPDB URL': os.path.join(base_path, f"SITE={entry}")})
    info.update({'Protein Mol2 Download URL': os.path.join(base_path, f'EXPORTENTRY=PROT={entry}')})
    info.update({'Ligand Mol2 Download URL': os.path.join(base_path, f'EXPORTENTRY=LIG={entry}')})
    info.update({'Site Mol2 Download URL': os.path.join(base_path, f'EXPORTENTRY=SITE={entry}')})
    info.update({'IFP Download URL': os.path.join(base_path, f'EXPORTENTRY=IFP={entry}')})
    info.update({'Cavity Download URL': os.path.join(base_path, f'EXPORTENTRY=CAVITY={entry}')})
    info.update({'Interaction Download URL': os.path.join(base_path, f'EXPORTENTRY=INTS={entry}')})
    info.update({'All files Download URL': os.path.join(base_path, f'EXPORTENTRY=ALL={entry}')})

    # Lastly, add information from SCOPe classification table:
    scope_sub_df = scope_df.loc[scope_df['pdb_id'] == info['PDB ID'], ["chain", "class"]]
    if len(scope_sub_df) == 0:
        info.update({'SCOPe Chain Classes': ''})
    else:
        scope_sub_df['combined'] = scope_sub_df[['chain', 'class']].apply(lambda row: row['chain'] + ' ' + row['class'], axis=1)
        chain_classes = " / ".join(scope_sub_df['combined'].to_list())
        # print(chain_classes)
        info.update({'SCOPe Chain Classes': chain_classes})
    
    return info

def parse_scPDB_pages(dir: str = 'Files/scPDB',
                      exist_ok: bool = False,
                      scope_path: str = 'Files/SCOPe/2_08.csv') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses source codes of scPDB weg pages and creates a dataframe. (Plus: a dataframe of entries which produced errors during parsing)

    Web pages source codes are expected to be already collected and stored in a pages_dir (by default - 'Data/Pages')
    The resulting dataframe will be stored as output_path (by default - 'Data/database.csv')
    Frame of errors will be stored in 'parsing_failed.csv'

    The function also needs a path to the SCOPe classification dataframe (by default - 'Data/SCOPe/2_08.csv')

    Args:
        dir (str, optional): directory with .txt files with source codes. Defaults to 'Data/Pages'.
        scope_path (str, optional): path to the SCOPe classification df. Defaults to Files/SCOPe/2_08.csv'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (final dataframe, dataframe of error logs)
    """

    # Check the input
    assert isinstance(dir, str), f'page_dir must be str, not {type(dir)}'
    assert isinstance(exist_ok, bool), f'exist_ok must be bool, not {type(exist_ok)}'
    assert isinstance(scope_path, str), f'scope_path must be str, not {type(scope_path)}'
    
    errors = []
    scope_df = pd.read_csv(scope_path)
    entries_found = 0
    parsings_skipped = 0
    htmls_found = 0
    htmls_parsed = 0

    # Report
    print('Parsing scPDB web pages')
    print(f'Entries provided (from {dir}): {len(os.listdir(dir))}')

    # Iterate through pages
    for entry in tqdm(os.listdir(dir), desc='Parsing Source Pages...', total=len(os.listdir(dir))):

        # Skip invalid folders
        if not len(entry.split('_')[0]) == 4  or not os.path.isdir(os.path.join(dir, entry)):
            continue

        entries_found += 1

        page = os.path.join(dir, entry, 'html.txt')
        html_path = os.path.join(dir, entry, 'html.json')

        # Skip if no html
        if not os.path.isfile(page):
            continue
        htmls_found += 1

        # Skip already parsed instances
        if os.path.isfile(html_path) and exist_ok:
            parsings_skipped += 1
            continue

        # Try reading htmls
        try:
            with open(page, 'r') as f:
                source_code = f.read()
            result = parse_scpdb_source_code(entry, source_code, scope_df)
            htmls_parsed += 1

            with open(html_path, 'w') as f:
                json.dump(result, f)
        
        except BaseException as e:

            # Register errors
            errors.append([entry, e])
            continue

    errors_df = pd.DataFrame(errors, columns=['entry', 'error'])

    # Report
    print(f'Entries found in the dir: {entries_found}')
    print(f'Htmls found: {htmls_found} ({100 * htmls_found / entries_found:.1f} %)')
    print(f'Skipped parsings: {parsings_skipped} ({100 * parsings_skipped / entries_found:.1f} %)')
    print(f'Successfully parsed: {htmls_parsed} ({100 * htmls_parsed / entries_found:.1f} %)')
    print(f'html.json now available for {parsings_skipped + htmls_parsed} entries ({100 * (parsings_skipped + htmls_parsed) / entries_found:.1f} %)')
    print(f'Errors occured: {len(errors_df)}')
    os.makedirs('Logs', exist_ok==True)
    errors_df.to_csv('Logs/failed_parsing.csv', sep='\t')

    return

def fetch_RCSB_metadata(exist_ok: bool = True,
                        dir: str = 'Files/scPDB',
                        exclude_entries: Collection = None):
    """
    Trying to fetch RCSB json files for all the entries from directory 'dir'.

    Args:
        exist_ok (bool, optional): skip already loaded files. Defaults to True.
        dir (str, optional): directory with entries to load. Defaults to 'Files/scPDB'.
        exclude_entries (Collection, optional): entries to exclude from the loading. Defaults to None.
    """

    def fetch_request(url: str):
        try:
            data = requests.get(url, timeout=5)
            time.sleep(random.randint(1, 10) / 10)
            return data
        except:
            return None
    
    # Check the input
    if exclude_entries:
        assert isinstance(exclude_entries, Collection), f'exclude_entries must be a Collection, not {type(exclude_entries)}'
    else:
        exclude_entries = set()
    assert isinstance(exist_ok, bool), f'exist_ok must be bool, not {type(exist_ok)}'
    assert isinstance(dir, str), f'dir must be str, not {type(dir)}'
    os.makedirs(dir, exist_ok=True)

    all_entries = sorted(set([file.replace('\n', '') for file in os.listdir(dir)]) - set(exclude_entries))

    # Report
    print('Fetching RCSB json files')
    print(f'Entries provided (from {dir}): {len(all_entries)}')
    print(f'Expected size of the downloads: {format_size(70_000 * len(all_entries))}')

    total_files = 0
    true_size = 0
    skipped_existing = 0
    failed_entries = []

    def check_json_exists(p: str):
        """
        Tool to check if a path to json exists, can be opened and has something inside
        """

        if not os.path.isfile(p):
            return False
        
        if os.path.getsize(p) < 100:
            return False
        
        try:
            json.load(open(p, 'rb'))
            return True
        except:
            return False


    with tqdm(total=len(all_entries), desc='Fetching RCSB jsons') as pbar:
        for entry in all_entries:
            # Iterate over entries

            pbar.set_description(f'Fetching RCSB jsons ({total_files} files, {format_size(true_size)})...')

            pdb_id = entry.split('_')[0]

            # Create the path and check if the main file 'rcsb_entry.json' already exists
            path = os.path.join(dir, entry, 'rcsb_entry.json')
            if exist_ok and check_json_exists(path):
                skipped_existing += 1
                entry_json = json.load(open(path, 'rb'))
            else:
                entry_url = f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}'
                entry_json = fetch_request(entry_url)
                if not entry_json or entry_json.status_code != 200:
                    failed_entries.append((entry, 'rcsb_entry.json'))
                    pbar.update(1)
                    continue
                else:
                    entry_json = dict(entry_json.json())
                with open(path, 'w') as handle:
                    json.dump(entry_json, handle)
            
            total_files += 1
            true_size += os.path.getsize(path)

            identifiers = entry_json["rcsb_entry_container_identifiers"]
            with open(os.path.join(dir, entry, 'rcsb_identifiers.json'), 'w') as handle:
                json.dump(identifiers, handle)
            
            total_files += 1
            true_size += os.path.getsize(os.path.join(dir, 'identifiers.json'))

            # Iteration through entities:
            for entity_id in identifiers['entity_ids']:

                if 'polymer_entity_ids' in identifiers and entity_id in identifiers['polymer_entity_ids']:
                    entity_group = 'polymer_entity'
                elif 'non_polymer_entity_ids' in identifiers and entity_id in identifiers['non_polymer_entity_ids']:
                    entity_group = 'nonpolymer_entity'
                elif 'branched_entity_ids' in identifiers and entity_id in identifiers['branched_entity_ids']:
                    entity_group = 'branched_entity'
                else:
                    continue
                
                path = os.path.join(dir, entry, f'rcsb_entity_{entity_id}.json')
                if exist_ok and check_json_exists(path):
                    skipped_existing += 1
                    entity_json = json.load(open(path, 'rb'))
                else:
                    entity_url = f'https://data.rcsb.org/rest/v1/core/{entity_group}/{pdb_id}/{entity_id}'
                    entity_json = fetch_request(entity_url)
                    if not entity_json or entity_json.status_code != 200:
                        failed_entries.append((entry, f'rcsb_entity_{entity_id}.json'))
                        continue
                    else:
                        entity_json = dict(entity_json.json())
                    with open(path, 'w') as handle:
                        json.dump(entity_json, handle)

                total_files += 1
                true_size += os.path.getsize(path)

                # Check chains for protein entity:
                if entity_group == 'polymer_entity':
                    
                    # Iterate through chains
                    for chain_id in entity_json["entity_poly"]['pdbx_strand_id'].split(','):

                        path = os.path.join(dir, entry, f'rcsb_entity_{entity_id}_chain_{chain_id}.json')
                        if exist_ok and check_json_exists(path):
                            skipped_existing += 1
                        else:
                            chain_url = f'https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{pdb_id}/{chain_id}'
                            chain_json = fetch_request(chain_url)
                            if not chain_json or chain_json.status_code != 200:
                                failed_entries.append((entry, f'rcsb_entity_{entity_id}_chain_{chain_id}.json'))
                                continue
                            else:
                                chain_json = dict(chain_json.json())
                            with open(path, 'w') as handle:
                                json.dump(chain_json, handle)

            # Fetch assemblies if any
            for assembly_id in identifiers['assembly_ids']:

                path = os.path.join(dir, entry, f'rcsb_assembly_{assembly_id}.json')
                if exist_ok and check_json_exists(path):
                    skipped_existing += 1
                else:
                    assembly_url = f'https://data.rcsb.org/rest/v1/core/assembly/{pdb_id}/{assembly_id}'
                    assembly_json = fetch_request(assembly_url)
                    if not assembly_json or assembly_json.status_code != 200:
                        failed_entries.append((entry, f'rcsb_assembly_{assembly_id}.json'))
                        continue
                    else:
                        assembly_json = dict(assembly_json.json())
                    with open(path, 'w') as handle:
                        json.dump(assembly_json, handle)

                total_files += 1
                true_size += os.path.getsize(path)
            
            pbar.update(1)

    failed_entries = pd.DataFrame(failed_entries, columns=['entry', 'file'])
    failed_entries.to_csv('Logs/failed_rcsb_fetches.csv', sep=',')
    print(f'Fetching finished (files fetched: {total_files}, actual size: {format_size(true_size)}; skipped as existing: {skipped_existing})')
    print(f'Failed entries: {len(failed_entries)}')

def map_pdb_to_uniprot(pdb_id: str | list[str],
                       mode: str = 'all',
                       per_chain: bool = True,
                       sifts_dir: str = 'SIFTS') -> dict[str, tuple] | list | str:
    """
    Maps PDB instance to UniProt AC based of SIFTS file

    Args:
        pdb_id (str): PDB id to map
        mode (str, optional): how may uniprots to return.
            - all: all uniprots found
            - first: only the first one found
            - longest: uniprot for the longest fragment
            Defaults to 'all'.
        per_chain (bool, optional): allows to return the result per chain (as a dict)
            or for the whole instance combined. Defaults to True.
        sifts_dir (str, optional): folder with SIFTS files. Defaults to 'SIFTS'.

    Returns:
        dict[str, tuple | str]: dictionary of uniprots per chain
        OR
        list: if per_chain is False and mode is "all"
        OR
        str: is per_chain is False and mode is "first" / "longest"
    """

    # Check the input
    if not isinstance(pdb_id, str | list):
        raise TypeError(f'pdb_id must be str | list, not {type(pdb_id)}')
    if isinstance(pdb_id, str):
        pdb_id = [pdb_id]
    for i, pdb_id_ in enumerate(pdb_id):
        pdb_id_ = ''.join([symbol for symbol in pdb_id_.lower() if symbol.isalnum()])
        if len(pdb_id_) != 4:
            raise ValueError(f'pdb_id must have 4 alphanumeric symbols, not: {pdb_id_}')
        pdb_id[i] = pdb_id_
    if not isinstance(mode, str):
        raise TypeError(f'mode must be str, not {type(mode)}')
    mode = mode.lower().strip()
    if mode not in ['all', 'first', 'longest']:
        raise ValueError(f'mode can be "all" / "first" / "longest", not: {mode}')
    if not isinstance(sifts_dir, str):
        raise TypeError(f'sifts_dir must be str, not {type(sifts_dir)}')
    if not isinstance(per_chain, bool):
        raise TypeError(f'per_chain must be bool, not {type(per_chain)}')

    # Get SIFTS mapping
    sifts = pd.read_csv(os.path.join(sifts_dir, 'pdb_chain_uniprot.csv'))

    # Find PDB matches
    uniprots = sifts.loc[sifts.PDB.isin(pdb_id)]
    uniprots = uniprots.drop_duplicates()
    
    all_results = {}
    for pdb_id_ in tqdm(pdb_id, desc='Fetching UniProts...', disable=(len(pdb_id) == 1)):

        all_results.update({pdb_id_: {}})
        uniprots_ =  uniprots[uniprots.PDB == pdb_id_]
        if len(uniprots_) == 0:
        
            all_results[pdb_id_] = None
            continue

        # Get chains
        if not per_chain:
            uniprots_['CHAIN'] = 'PLACEHOLDER'
        chains = uniprots_.CHAIN.unique()

        # Collect UniProts per chain based on the mode
        for chain in chains:
            uniprots_of_chain = uniprots_.loc[uniprots_.CHAIN == chain]

            if mode == 'first':

                all_results[pdb_id_].update({chain: uniprots_of_chain.iloc[0]['SP_PRIMARY']})
                continue
            

            codes = uniprots_of_chain["SP_PRIMARY"].to_list()
            if mode == 'all':
                all_results[pdb_id_].update({chain: sorted(set(codes))})
                continue
            
            uniprots_of_chain['Length'] = uniprots_of_chain["RES_END"] - uniprots_of_chain["RES_BEG"]
            uniprots_of_chain = uniprots_of_chain.loc[uniprots_of_chain['Length'] == uniprots_of_chain['Length'].max()]
            longest = uniprots_of_chain.iloc[0]["SP_PRIMARY"]
            all_results[pdb_id_].update({chain: longest})

        if not per_chain:
            all_results[pdb_id_] = all_results[pdb_id_]['PLACEHOLDER']

    if len(pdb_id) == 1:
            return all_results[pdb_id[0]]
    return all_results

def map_pdb_to_ec_number(pdb_id: str,
                         mode: str = 'all',
                         per_chain: bool = True,
                         sifts_dir: str = 'SIFTS') -> dict[str, tuple] | list | str:
    """
    Maps PDB instance to EC number based of SIFTS file

    Args:
        pdb_id (str): PDB id to map
        mode (str, optional): how may EC numbers to return.
            - all: all uniprots found
            - first: only the first one found
            Defaults to 'all'.
        per_chain (bool, optional): allows to return the result per chain (as a dict)
            or for the whole instance combined. Defaults to True.
        sifts_dir (str, optional): folder with SIFTS files. Defaults to 'SIFTS'.

    Returns:
        dict[str, tuple | str]: dictionary of EC numbers per chain
        OR
        list: if per_chain is False and mode is "all"
        OR
        str: is per_chain is False and mode is "first"
    """

    # Check the input
    if not isinstance(pdb_id, str):
        raise TypeError(f'pdb_id must be str, not {type(pdb_id)}')
    pdb_id = ''.join([symbol for symbol in pdb_id.lower() if symbol.isalnum()])
    if len(pdb_id) != 4:
        raise ValueError(f'pdb_id must have 4 alphanumeric symbols, not: {pdb_id}')
    if not isinstance(mode, str):
        raise TypeError(f'mode must be str, not {type(mode)}')
    mode = mode.lower().strip()
    if mode not in ['all', 'first']:
        raise ValueError(f'mode can be "all" / "first" / "longest", not: {mode}')
    if not isinstance(sifts_dir, str):
        raise TypeError(f'sifts_dir must be str, not {type(sifts_dir)}')
    if not isinstance(per_chain, bool):
        raise TypeError(f'per_chain must be bool, not {type(per_chain)}')

    # Get SIFTS mapping
    sifts = pd.read_csv(os.path.join(sifts_dir, 'pdb_chain_enzyme.csv'))

    # Find PDB matches
    ecs = sifts.loc[sifts.PDB == pdb_id]
    ecs = ecs.drop_duplicates()
    if len(ecs) == 0:
        return None

    # Get Chains
    if not per_chain:
        ecs['CHAIN'] = 'PLACEHOLDER'
    chains = ecs.CHAIN.unique()

    # Collect EC numbers per chain based on the mode
    results = {}
    for chain in chains:
        ecs_of_chain = ecs.loc[ecs.CHAIN == chain]

        if mode == 'first':

            results.update({chain: ecs_of_chain.iloc[0]['EC_NUMBER']})
            continue

        codes = ecs_of_chain["EC_NUMBER"].to_list()
        if mode == 'all':
            results.update({chain: sorted(set(codes))})

    if not per_chain:
        return results['PLACEHOLDER']
    return results

def map_scop_node_to_name(node_id: int, csv: str = 'SIFTS/scop_names.csv'):

    scop_names = pd.read_csv(csv)
    name = scop_names.loc[scop_names['Code'].astype(int) == node_id, 'Description'].to_list()

    if not name:
        return None
    return name[0]

def map_pdb_to_scop(pdb_id: str,
                    level: str | int = "protein type",
                    mode: str = 'all',
                    per_chain: bool = True,
                    as_names: bool = True,
                    scop_dir: str = 'SIFTS') -> dict[str, tuple] | list | str:
    """
    Maps PDB instance to SCOP classification based of SIFTS file

    Args:
        pdb_id (str): PDB id to map
        level (str | int, optional): what level of classification to extract.
            There are 5 levels: 'protein type', 'protein class', 'fold', 'superfamily', 'family'.
            They can be chosen by name or by index from 0 to 4 (included). Defaults to 'protein type'
        mode (str, optional): how may identifiers to return.
            - all: all identifiers found
            - first: only the first one found
            - longest: identifier of the longest fragment
            Defaults to 'all'.
        per_chain (bool, optional): allows to return the result per chain (as a dict)
            or for the whole instance combined. Defaults to True.
        as_names (bool, optional): allows to return textual descriptions of identifiers
            instead of SCOP node codes. Defaults to True.
        scop_dir (str, optional): folder with SIFTS files. Defaults to 'SIFTS'.

    Returns:
        dict[str, tuple | str]: dictionary of identifiers per chain
        OR
        list: if per_chain is False and mode is "all"
        OR
        str: is per_chain is False and mode is "first"
    """

    # Check the input
    if not isinstance(pdb_id, str):
        raise TypeError(f'pdb_id must be str, not {type(pdb_id)}')
    pdb_id = ''.join([symbol for symbol in pdb_id.lower() if symbol.isalnum()])
    if len(pdb_id) != 4:
        raise ValueError(f'pdb_id must have 4 alphanumeric symbols, not: {pdb_id}')
    if not isinstance(mode, str):
        raise TypeError(f'mode must be str, not {type(mode)}')
    mode = mode.lower().strip()
    if mode not in ['all', 'first', 'longest']:
        raise ValueError(f'mode can be "all" / "first" / "longest", not: {mode}')
    if not isinstance(scop_dir, str):
        raise TypeError(f'sifts_dir must be str, not {type(scop_dir)}')
    
    allowed_levels = ['protein type', 'protein class', 'fold', 'superfamily', 'family']
    if not isinstance(level, str | int):
        raise TypeError(f'level must be str or int, not {type(level)}')
    if isinstance(level, int) and level not in list(range(5)):
        raise ValueError(f'integer lavel can be 0, 1, 2, 3 or 4, not {level}')
    if isinstance(level, int):
        level = allowed_levels[level]
    else:
        level = level.lower().strip().replace('_', ' ')
        if level not in allowed_levels:
            raise ValueError(f'level can be one of {allowed_levels}, not: {level}')
    if not isinstance(as_names, bool):
        raise TypeError(f'as_names must be bool, not {type(as_names)}')
    if not isinstance(per_chain, bool):
        raise TypeError(f'per_chain must be bool, not {type(per_chain)}')

    # Get SIFTS mapping
    scop = pd.read_csv(os.path.join(scop_dir, 'scop.csv'))
    
    # Get matching PDBs
    scop = scop.loc[scop["FA-PDBID"].str.lower() == pdb_id, ["FA-PDBREG", level]]
    if len(scop) == 0:
        return None
    
    # Get Chains
    scop[['CHAIN', 'Span']] = scop["FA-PDBREG"].str.split(':', n=1, expand=True)
    if not per_chain:
        scop['CHAIN'] = 'PLACEHOLDER'
    scop = scop.drop_duplicates()
    chains = scop.CHAIN.unique()
    scop['names'] = scop[level].apply(map_scop_node_to_name)

    # Collect UniProts per chain based on the mode
    results = {}
    for chain in chains:
        scop_of_chain = scop.loc[scop.CHAIN == chain]

        if mode == 'first':

            results.update({chain: scop_of_chain.iloc[0][level if not as_names else 'names']})
            continue
        

        codes = scop_of_chain[level if not as_names else 'names'].to_list()
        if mode == 'all':
            results.update({chain: sorted(set(codes))})
            continue
        
        scop_of_chain['Span'] = '(' + scop_of_chain['Span'].str.replace(r',[A-Z0-9]:', ')+(', regex=True) + ')'
        scop_of_chain['Length'] = scop_of_chain['Span'].apply(eval).abs()
        scop_of_chain = scop_of_chain.loc[scop_of_chain['Length'] == scop_of_chain['Length'].max()]
        longest = scop_of_chain.iloc[0][level if not as_names else 'names']
        results.update({chain: longest})

    if not per_chain:
        return results['PLACEHOLDER']
    return results

def map_uniprot_to_uniref(uniprot: str,
                          level: int = 50,
                          mode: str = 'all',
                          add_cluster_names: bool = True) -> list[tuple | str] | tuple | str:
    """
    Maps UniProt instance to UniRef cluster using API

    Args:
        uniprot (str): UniProt id to map
        level (int, optional): what level of similarity to map.
            There are 3 levels: 50, 90 and 100. Defaults to 50
        mode (str, optional): how may clusters to return.
            - all: all clusters found
            - first: only the first one found
            - largest: return the largest cluster
            Defaults to 'all'.
        add_cluster_names (bool, optional): allows to also return the names of clusters. Defaults to True.

    Returns:
        list[tuple]: if mode is 'all' and add_cluster_names is True
        OR
        tuple: if mode is "first" / "largest" and add_cluster_names is True
        OR
        str: if mode is "first" / "largest" and add_cluster_names is False
    """

    # Check the input
    if not isinstance(uniprot, str):
        raise TypeError(f'uniprot must be str, not {type(uniprot)}')
    if not isinstance(mode, str):
        raise TypeError(f'mode must be str, not {type(mode)}')
    mode = mode.lower().strip()
    if mode not in ['all', 'first', 'largest']:
        raise ValueError(f'mode can be "all" / "first" / "largest", not: {mode}')
    
    allowed_levels = [50, 90, 100]
    if not isinstance(level, int):
        raise TypeError(f'level must be int, not {type(level)}')
    if level not in allowed_levels:
        raise ValueError(f'level can be one of {allowed_levels}, not: {level}')
    if not isinstance(add_cluster_names, bool):
        raise TypeError(f'add_cluster_names must be bool, not {type(add_cluster_names)}')

    # Get respose
    url = f'https://rest.uniprot.org/uniref/search?format=json&query=%28uniprot_id%3A{uniprot}+AND+identity%3A{round(level/100, 1)}%29&size={1 if mode == "first" else 10}'
    response = requests.get(url, json=True)

    if response.status_code != 200:
        return None
    
    js = response.json()['results']

    if mode == 'first':

        js = js[0]
        cluster_id = js['id']
        cluster_name = js['name']

        return cluster_id, cluster_name
    
    results = []
    for cluster in js:
        results.append((cluster['id'], cluster['name'], cluster['memberCount']))

    if mode == 'all':
        return [(r[0], r[1]) for r in results]
    
    if mode == 'largest':

        largest = sorted(results, key=lambda r: r[-1], reverse=True)[0]
        return largest[0], largest[1]

def get_uniprot_json(uniprot: str,
                     dir_to_save: str | None = None,
                     filename: str = 'uniprot.json'):
    """
    Extracts JSON using UniProt REST API
    If dir_to_save is not None, saves 'uniprot.json' into this directory.

    Args:
        uniprot (str): UniProt id to query
        dir_to_save (str, optional): folder to save the result. If None, no saving. Defaults to None.
        filename (str, optional): filename to save the result. Defaults to 'uniprot.json'.

    Returns:
        dict
    """

    # Check the input
    if not isinstance(uniprot, str):
        raise TypeError(f'uniprot must be str, not {type(uniprot)}')
    if dir_to_save is not None:
        if not isinstance(dir_to_save, str):
            raise TypeError(f'dir_to_save must be str, not {type(dir_to_save)}')
        if not isinstance(filename, str):
            raise TypeError(f'filename must be str, not {type(filename)}')
    
    # Get query
    url = f'https://www.uniprot.org/uniprotkb/{uniprot.upper().strip()}.json'
    response = requests.get(url, json=True, timeout=5)

    if response.status_code != 200:
        return {}
    else:
        result = response.json()

    if dir_to_save is not None:
        with open(os.path.join(dir_to_save, filename), 'w') as f:
            json.dump(result, f)
    
    return result

def fetch_uniprot_metadata(exist_ok: bool = True,
                           dir: str = 'Files/scPDB',
                           sifts_dir: str = 'SIFTS',
                           exclude_entries: Collection = None):
    """
    Iterate over the provided directory (dir), try to find UniProt's
    corresponding to the entry, fetch the API and store JSONs

    Args:
        exist_ok (bool, optional): skip if JSONs are already there. Defaults to True.
        dir (str, optional): directory of entries. Defaults to 'Files/scPDB'.
        sifts_ir (str, optional): directory with SIFTS files. Defaults to 'SIFTS'.
        exclude_entries (Collection, optional): entries to skip. Defaults to None.
    """

    def fetch_request(uniprot: str, folder: str):
        try:
            data = get_uniprot_json(
                uniprot=uniprot,
                dir_to_save=folder,
                filename=f'{uniprot}.json')
            time.sleep(random.randint(1, 10) / 10)
            return data
        except:
            return None
    
    # Check the input
    if exclude_entries:
        assert isinstance(exclude_entries, Collection), f'exclude_entries must be a Collection, not {type(exclude_entries)}'
    else:
        exclude_entries = set()
    assert isinstance(exist_ok, bool), f'exist_ok must be bool, not {type(exist_ok)}'
    assert isinstance(dir, str), f'dir must be str, not {type(dir)}'

    all_entries = sorted(set([file.replace('\n', '') for file in os.listdir(dir)
                              if file.replace('\n', '')[0] != '.' and len(file.replace('\n', '')) <= 7]) - set(exclude_entries))

    # Report
    print('Fetching UniProt json files')
    print(f'Entries provided (from {dir}): {len(all_entries)}')
    print(f'Expected size of the downloads: {format_size(100_000 * len(all_entries))}') # ???

    total_files = 0
    total_loaded = 0
    true_size = 0
    loaded_size = 0
    skipped_existing = 0
    failed_entries = []

    def check_json_exists(p):

        if not os.path.isfile(p):
            return False
        
        if os.path.getsize(p) < 100:
            return False

        
        try:
            json.load(open(p, 'rb'))
            return True
        except:
            return False
        
    all_uniprots = map_pdb_to_uniprot(
                        pdb_id=[entry.split('_')[0] for entry in all_entries],
                        mode='all',
                        per_chain=True,
                        sifts_dir=sifts_dir)

    major_uniprots = map_pdb_to_uniprot(
                pdb_id=[entry.split('_')[0] for entry in all_entries],
                mode='longest',
                per_chain=False,
                sifts_dir=sifts_dir)


    with tqdm(total=len(all_entries), desc='Fetching UniProt JSONs') as pbar:
        for entry in all_entries:

            pbar.set_description(f'Fetching UniProt JSONs (All: {total_files} files, {format_size(true_size)}, Loaded:  {total_loaded} files, {format_size(loaded_size)})...')

            pdb_id = entry.split('_')[0]

            # Extract UniProts using SIFTS
            # uniprots = map_pdb_to_uniprot(
            #     pdb_id=pdb_id,
            #     mode='all',
            #     per_chain=True,
            #     sifts_dir=sifts_dir)
            uniprots = all_uniprots[pdb_id.lower()]
            if not uniprots:
                failed_entries.append((entry, 'UniProt Not Found'))
                pbar.update(1)
                continue

            # Extract major protein UniProt
            # major_uniprot = map_pdb_to_uniprot(
            #     pdb_id=pdb_id,
            #     mode='longest',
            #     per_chain=False,
            #     sifts_dir=sifts_dir)
            major_uniprot = major_uniprots[pdb_id.lower()]

            already_collected_uniprots = set()

            # Iterate over chains
            for chain, ups in uniprots.items():
                
                # Iterate over codes
                for up in ups:

                    if up in already_collected_uniprots:
                        continue
                    else:
                        already_collected_uniprots.add(up)

                    # Create the path and check if the main file '{UniProt}.json' already exists
                    path = os.path.join(dir, entry, f'{up}.json')
                    if exist_ok and check_json_exists(path):
                        skipped_existing += 1
                    else:
                        entry_json = fetch_request(uniprot=up, folder=os.path.join(dir, entry))
                        if not entry_json:
                            failed_entries.append((entry, f'{up} failed to fetch'))
                            pbar.update(1)
                            continue
                        else:
                            entry_json = dict(entry_json)
                            entry_json.update({'Chain': chain, 'IsMain': up.upper() == major_uniprot.upper()})
                        with open(path, 'w') as handle:
                            json.dump(entry_json, handle)
                        total_loaded += 1
                        loaded_size += os.path.getsize(path)
                
                    total_files += 1
                    true_size += os.path.getsize(path)
            pbar.update(1)


    failed_entries = pd.DataFrame(failed_entries, columns=['entry', 'file'])
    failed_entries.to_csv('Logs/failed_uniprot_fetches.csv', sep=',')
    print(f'Fetching finished (files fetched: {total_files}, actual size: {format_size(true_size)}; skipped as existing: {skipped_existing})')
    print(f'Failed entries: {len(failed_entries)}')
