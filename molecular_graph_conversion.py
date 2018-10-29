import gzip
import os
import sys


def read_mol2_file(mol2_filename, database_smiles, sizes, out_directory):

    # read in mol2 file
    mol2_file = gzip.open(mol2_filename, 'rb')
    molecules = mol2_file.read().decode().split("@<TRIPOS>MOLECULE")

    # iterate through molecules
    for molecule in molecules[1:]:

        # print "Debug mol", molecule
        name = molecule.split("@<TRIPOS>BOND")[0].split('\n')[1].strip()

        # print name
        size = int(molecule.split("@<TRIPOS>BOND")[
                   0].split('\n')[2].strip().split("    ")[0])
        bonds = molecule.split("@<TRIPOS>BOND")[1].strip().split('\n')
        flag = False
        for bond in bonds:
            el = bond.split("   ")
            bt = el[-1].split()[-1]
            if bt == 'ar' or bt == 'am' or bt == 'du' or bt == 'un' or bt == 'nc':
                #count += 1
                flag = True
                break

        if not flag:

            # if size in range(27,56) or size == 57:
            if size in sizes:
                smiles = database_smiles[name]
                for bond in bonds:
                    tokens = bond.split()
                    with open(os.path.join(out_directory,"n_{}/{}.txt".format(size, name)), "a") as fw:
                        fw.write(
                            str(int(tokens[1])-1)+" "+str(int(tokens[2])-1) + " {" + tokens[3] + "}\n")

                with open(os.path.join(out_directory,"n_{}/smiles.txt".format(size)), "a") as fw:
                    fw.write(smiles+"\n")

# data locations
data_folder = "/home/link15/wip/graphvae/data/zinc12_druglike"
smi_filename = "/home/link15/wip/graphvae/data/zinc12_druglike/13_p0.smi"
mol2_filenames = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if '.mol2.gz' in f]

# set sizes allowed
sizes = list(range(30, 41))
out_directory = '/home/link15/wip/graphvae/zinc/training/'
for size in sizes:
    new_path = os.path.join(out_directory, "n_{}".format(size))
    if not os.path.isdir(new_path):
        os.makedirs(new_path)

# read in SMILES and their IDs
database_smiles = {}
smi_file = open(smi_filename)
for line in smi_file:
    [smiles, idzinc] = line.strip().split(" ")
    database_smiles[idzinc] = smiles

for mol2_filename in mol2_filenames:
    print(mol2_filename)
    read_mol2_file(mol2_filename, database_smiles, sizes, out_directory)