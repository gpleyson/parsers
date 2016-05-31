import os
import numpy as np
import pandas as pd
import tables as tb
import xml.etree.cElementTree as ET


class VasprunParser(object):
    """
        Class for parsing vasprun.xml and querying hdf5 files generated
        from vasprun.xml.
        *** DEPRECATED ***
        This class has been separated into two classes; one to parse xml
        and convert it to h5, and one to load and query the h5 file
    """
    def __init__(self):
        self.dict_vasprun = {}

    def parse_vasprun(self, directory='.', filename='vasprun.xml'):
        """
        Create a dictionary representation of vasprun.xml

        :params:
            directory - directory where the xml file is located
            filename - filename to be parsed

        :return:
            self.dict_vasprun - dictionary representation of the
                vasprun.xml
        """
        path = os.path.join(directory, filename)
        root = ET.parse(path).getroot()
        try:
            self.dict_vasprun = parse_node(root)
        except:
            print('vasprun.xml not found in ', path)

    def to_h5(self, directory='.', filename='vasprun.h5',
              root='/', mode='w', title='vasprun'):
        """
        Stores self.dict_vasprun to h5 file format.

        :params:
            directory - Directory where the h5 file is to be stored.
                (default: '.')
            filename - Filename of h5 file. (default: 'vasprun.h5')
            root - Root location of the vasprun data in the h5 file.
                (default: '/')
            mode - Writing mode (default: 'w')
            title - Title of the vasprun node. (default: 'vasprun')
        """
        path = directory + '/' + filename
        FILTERS = tb.Filters(complib='zlib', complevel=9)
        h5file = tb.open_file(path, mode=mode, title=title,
                              filters=FILTERS)
        store_to_h5(h5file, root, self.dict_vasprun)
        h5file.flush()
        h5file.close()

    def load_h5(self, directory='.', filename='vasprun.h5',
                root='/', mode='r'):
        """
        Loads an hdf5 file for parsing.

        :params:
            directory - Directory where h5 file is located. (default: 'w')
            filename - Fielname of h5 file. (default: 'vasprun.h5')
            root - Root location of the vasprun data in the h5 file.
                (default: '/')
            mode - Writing mode (default: 'r')
        """
        path = directory + '/' + filename
        self.h5file = tb.open_file(path, mode=mode)
        self.root = self.h5file.get_node(root)

    def _get_series(self, node):
        """
        Returns a pandas series representation of the leaves
        of a node in an h5 file.

        :params:
            node - Node to be parsed.

        :return:
            pd.Series - Pandas series representation of the node leaves.
        """
        return pd.Series({self.h5file.get_node(node, leaf)._v_name:
                          self.h5file.get_node(node, leaf).read()
                          for leaf in node._v_leaves})

    def get_generator(self):
        """
        Gets pandas series of generator paramters.

        :return:
            pd.Series
        """
        return self._get_series(self.root.generator)

    def get_incar(self):
        """
        Gets pandas series of INCAR paramters.

        :return:
            pd.Series
        """
        return self._get_series(self.root.incar)

    def get_atoms(self):
        """
        Returns pandas dataframe of ./atomsinfo/atoms

        :return:
            pd.DataFrame
        """
        atomtype = self.root.atominfo.atoms.data.atomtype.read()
        element = self.root.atominfo.atoms.data.element.read()
        df = pd.DataFrame([element, atomtype], index=['element', 'atomtype'])
        return df.T

    def get_atomtypes(self):
        """
        Returns pandas dataframe of ./atomsinfo/atomtypes

        :return:
            pd.DataFrame
        """
        atomspertype = self.root.atominfo.atomtypes.data.atomspertype.read()
        mass = self.root.atominfo.atomtypes.data.mass.read()
        pseudopotential = \
            self.root.atominfo.atomtypes.data.pseudopotential.read()
        valence = self.root.atominfo.atomtypes.data.valence.read()
        element = self.root.atominfo.atomtypes.data.element.read()

        df = pd.DataFrame({'atomspertype': atomspertype,
                           'pseudopotential': pseudopotential,
                           'valence': valence,
                           'mass': mass}, index=element)
        return df

    def get_positions(self, state='final', coords='cartesian'):
        """
        Returns a pandas dataframe of the initial or final
        atomic positions.
        Returns cartesian coordinates of coords='cartesian'
        or relative coordinates if coords='relative'.

        :params:
            coords - 'cartesian' or 'relative'
            state = 'initial' for initial position and
                    'final' for final position

        :return:
            pd.DataFrame
        """
        nodename = state + 'pos'
        node = self.h5file.get_node(self.root, nodename)
        dfelement = self.get_atoms()['element']
        relpos = node.positions.read()
        if coords == 'cartesian':
            basis = node.crystal.basis.read()
            positions = np.dot(relpos, basis)
            dfpos = pd.DataFrame(positions, columns=['cx', 'cy', 'cz'])
        elif coords == 'relative':
            dfpos = pd.DataFrame(relpos, columns=['rx', 'ry', 'rz'])

        return pd.concat([dfelement, dfpos], axis=1)

    def get_volume(self, state='final'):
        """
        Returns the volume.

        :params:
        state = 'initial' for initial position and
                'final' for final position

        :return:
            pd.Series
        """
        nodename = state + 'pos'
        node = self.h5file.get_node(self.root, nodename)

        return pd.Series(node.crystal.volume.read(), index=['volume'])

    def get_energies(self):
        """
        Returns pandas dataframe of the energies after each
        ionic step.

        :return:
            pd.DataFrame
        """
        node = self.h5file.get_node(self.root, 'calculation')
        df_en = pd.DataFrame()
        for child in node._f_iter_nodes():
            iteren = child.energy._f_iter_nodes()
            dictn = {entype._v_name: entype.read() for entype in iteren}
            df_en = df_en.append(pd.Series(dictn), ignore_index=True)

        return df_en

    def get_kpt_division(self):
        """
        Returns pandas series of the k-point divisions.

        :return:
            pd.Series
        """
        if 'Monkhorst_Pack' in self.root.kpoints._v_children:
            node = self.root.kpoints.Monkhorst_Pack.divisions
        elif 'Gamma' in self.root.kpoints._v_children:
            node = self.root.kpoints.Gamma.divisions
        else:
            print('Cannot get kpoint division')
        kpt_division = node.read()

        return pd.Series(kpt_division, index=['kx', 'ky', 'kz'])

    def get_cell_vectors(self):
        """
        Returns a 3x3 numpy array of the cell vectors.

        :return:
            np.array - 3x3 numpy array of cell vectors
        """
        return np.array(self.root.initialpos.crystal.basis.read())


class VasprunHDFParser(object):
    """
        Class for querying vasprun hdf5 files generated
        from vasprun.xml
    """
    def __init__(self, directory='.', filename='vasprun.h5',
                 root='/', mode='r'):
        """
            :params:
                directory - Directory where h5 file is located. (default: 'w')
                filename - Fielname of h5 file. (default: 'vasprun.h5')
                root - Root location of the vasprun data in the h5 file.
                    (default: '/')
                mode - Writing mode (default: 'r')
        """
        self.directory = directory
        self.filename = filename
        self.root_name = root
        self.mode = mode

    def __enter__(self):
        self.load_h5()
        return self

    def __exit__(self, type, value, traceback):
        self.h5file.close()

    def load_h5(self):
        """
            Loads an hdf5 file for parsing.
        """
        path = os.path.join(self.directory, self.filename)
        self.h5file = tb.open_file(path, mode=self.mode)
        self.root = self.h5file.get_node(self.root_name)

    def _get_series(self, node):
        """
            Returns a pandas series representation of the leaves
            of a node in an h5 file.

            :params:
                node - Node to be parsed.

            :return:
                pd.Series - Pandas series representation of the node leaves.
        """
        return pd.Series({self.h5file.get_node(node, leaf)._v_name:
                          self.h5file.get_node(node, leaf).read()
                          for leaf in node._v_leaves})

    def get_generator(self):
        """
            Gets pandas series of generator paramters.

            :return:
                pd.Series
        """
        return self._get_series(self.root.generator)

    def get_incar(self):
        """
            Gets pandas series of INCAR paramters.

            :return:
                pd.Series
        """
        return self._get_series(self.root.incar)

    def get_atoms(self):
        """
            Returns pandas dataframe of ./atomsinfo/atoms

            :return:
                pd.DataFrame
        """
        atomtype = self.root.atominfo.atoms.data.atomtype.read()
        element = self.root.atominfo.atoms.data.element.read()
        df = pd.DataFrame([element, atomtype], index=['element', 'atomtype'])
        return df.T

    def get_atomtypes(self):
        """
            Returns pandas dataframe of ./atomsinfo/atomtypes

            :return:
                pd.DataFrame
        """
        atomspertype = self.root.atominfo.atomtypes.data.atomspertype.read()
        mass = self.root.atominfo.atomtypes.data.mass.read()
        pseudopotential = \
            self.root.atominfo.atomtypes.data.pseudopotential.read()
        valence = self.root.atominfo.atomtypes.data.valence.read()
        element = self.root.atominfo.atomtypes.data.element.read()

        df = pd.DataFrame({'atomspertype': atomspertype,
                           'pseudopotential': pseudopotential,
                           'valence': valence,
                           'mass': mass}, index=element)
        return df

    def get_positions(self, state='final', coords='cartesian'):
        """
            Returns a pandas dataframe of the initial or final
            atomic positions.
            Returns cartesian coordinates of coords='cartesian'
            or relative coordinates if coords='relative'.

            :params:
                coords - 'cartesian' or 'relative'
                state = 'initial' for initial position and
                        'final' for final position

            :return:
                pd.DataFrame
        """
        nodename = state + 'pos'
        node = self.h5file.get_node(self.root, nodename)
        dfelement = self.get_atoms()['element']
        relpos = node.positions.read()
        if coords == 'cartesian':
            basis = node.crystal.basis.read()
            positions = np.dot(relpos, basis)
            dfpos = pd.DataFrame(positions, columns=['cx', 'cy', 'cz'])
        elif coords == 'relative':
            dfpos = pd.DataFrame(relpos, columns=['rx', 'ry', 'rz'])

        return pd.concat([dfelement, dfpos], axis=1)

    def get_volume(self, state='final'):
        """
            Returns the volume.

            :params:
            state = 'initial' for initial position and
                    'final' for final position

            :return:
                pd.Series
        """
        nodename = state + 'pos'
        node = self.h5file.get_node(self.root, nodename)

        return pd.Series(node.crystal.volume.read(), index=['volume'])

    def get_energies(self):
        """
            Returns pandas dataframe of the energies after each
            ionic step.

            :return:
                pd.DataFrame
        """
        node = self.h5file.get_node(self.root, 'calculation')
        df_en = pd.DataFrame()
        for child in node._f_iter_nodes():
            iteren = child.energy._f_iter_nodes()
            dictn = {entype._v_name: entype.read() for entype in iteren}
            df_en = df_en.append(pd.Series(dictn), ignore_index=True)

        return df_en

    def get_kpt_division(self):
        """
            Returns pandas series of the k-point divisions.

            :return:
                pd.Series
        """
        if 'Monkhorst_Pack' in self.root.kpoints._v_children:
            node = self.root.kpoints.Monkhorst_Pack.divisions
        elif 'Gamma' in self.root.kpoints._v_children:
            node = self.root.kpoints.Gamma.divisions
        else:
            print('Cannot get kpoint division')
        kpt_division = node.read()

        return pd.Series(kpt_division, index=['kx', 'ky', 'kz'])

    def get_cell_vectors(self):
        """
            Returns a 3x3 numpy array of the cell vectors.

            :return:
                np.array - 3x3 numpy array of cell vectors
        """
        return np.array(self.root.initialpos.crystal.basis.read())


class VasprunXMLParser(object):
    """
        Class for parsing vasprun.xml and and storing it to an h5 file.
    """
    def __init__(self, directory='.', filename='vasprun.xml'):
        """
            :params:
                directory - directory where the xml file is located
                filename - filename to be parsed
        """
        self.dict_vasprun = {}
        self.directory = directory
        self.filename = filename

    def parse_vasprun(self):
        """
            Create a dictionary representation of vasprun.xml

            :return:
                self.dict_vasprun - dictionary representation of the
                    vasprun.xml
        """
        path = os.join(self.directory, self.filename)
        root = ET.parse(path).getroot()
        try:
            self.dict_vasprun = parse_node(root)
        except:
            print('vasprun.xml not found in ', path)

    def to_h5(self, directory='.', filename='vasprun.h5',
              root='/', mode='w', title='vasprun'):
        """
            Stores self.dict_vasprun to h5 file format.

            :params:
                directory - Directory where the h5 file is to be stored.
                    (default: '.')
                filename - Filename of h5 file. (default: 'vasprun.h5')
                root - Root location of the vasprun data in the h5 file.
                    (default: '/')
                mode - Writing mode (default: 'w')
                title - Title of the vasprun node. (default: 'vasprun')
        """
        path = directory + '/' + filename
        FILTERS = tb.Filters(complib='zlib', complevel=9)
        h5file = tb.open_file(path, mode=mode, title=title,
                              filters=FILTERS)
        store_to_h5(h5file, root, self.dict_vasprun)
        h5file.flush()
        h5file.close()


def clean_key(key):
    """
    Replace white spaces and dashes in key with undersores.

    :params:
        key - String to be cleaned.

    :return:
        string - String with dashes and whitespaces replaced with _.
    """
    return key.replace(' ', '_').replace('-', '_')


def store_to_h5(h5file, parent, dict_nodes):
    """
    Converts dict_nodes into an hdf5 file.

    :params:
        h5file - hdf5 file handle
        parent - parent node in the hdf5file
        dict_nodes = dictionary representation of data
                        (see parse_node)
    """
    for key in dict_nodes.keys():
        if isinstance(dict_nodes[key], dict):
            node = h5file.create_group(parent, clean_key(key))
            store_to_h5(h5file, node, dict_nodes[key])
        elif isinstance(dict_nodes[key], list):
            node = h5file.create_group(parent, clean_key(key))
            for ii, value in enumerate(dict_nodes[key]):
                newkey = clean_key(key) + str(ii).zfill(4)
                newdict = {newkey: value}
                store_to_h5(h5file, node, newdict)
        else:
            h5file.create_array(parent, key.replace(' ', '_'),
                                dict_nodes[key], byteorder='little')


def parse_node(node):
    """
    Parse nodes from vasprun.xml.

    :params:
        node - cElementTree node from vasprun.xml

    :return:
        data - dictionary representation of the node.
    """
    data = {}
    for leaf in node:
        if leaf.tag == 'i':
            data.update(parse_item(leaf))
        elif leaf.tag == 'v':
            data.update(parse_vector(leaf))
        elif leaf.tag == 'varray':
            data.update(parse_varray(leaf))
        elif leaf.tag == 'separator':
            data.update(parse_separator(leaf))
        elif leaf.tag == 'atoms':
            data.update(parse_atoms(leaf))
        elif leaf.tag == 'types':
            data.update(parse_types(leaf))
        elif leaf.tag == 'array':
            data.update(parse_array(leaf))
        elif leaf.tag == 'time':
            data.update(parse_vector(leaf))
        elif leaf.tag == 'eigenvalues':
            data.update(parse_eigenvectors(leaf))
        elif leaf.tag == 'crystal':
            data.update(parse_generic(leaf))
        elif leaf.tag == 'generation':
            data.update(parse_generic(leaf))
        elif leaf.tag == 'energy':
            data.update(parse_generic(leaf))
        elif leaf.tag == 'structure':
            data.update(parse_generic(leaf))
        elif leaf.tag == 'generator':
            data.update(parse_generic(leaf))
        elif leaf.tag == 'incar':
            data.update(parse_generic(leaf))
        elif leaf.tag == 'kpoints':
            data.update(parse_generic(leaf))
        elif leaf.tag == 'parameters':
            data.update(parse_generic(leaf))
        elif leaf.tag == 'atominfo':
            data.update(parse_generic(leaf))
        elif (leaf.tag == 'scstep') | (leaf.tag == 'calculation'):
            if leaf.tag in data.keys():
                data[leaf.tag].append(parse_generic(leaf)[leaf.tag])
            else:
                data.update(parse_generic(leaf))
                data[leaf.tag] = [data[leaf.tag]]
        else:
            print('Unkown tag: ', leaf.tag)
            print('Will try to parse it anyway. Wish me luck!')
            try:
                data.update(parse_generic(leaf))
            except:
                pass

    return data


def parse_item(item):
    """
    Parse item leaf in vasprun.xml.

    :params:
        item - item node from vasprun.xml

    :return:
        dict - dictionary representation of the node.
    """
    dict_attrib = item.attrib

    if 'name' in dict_attrib:
        keyname = dict_attrib['name']
    else:
        print('No name found in item node')
        raise KeyError

    if 'type' in dict_attrib:
        datatype = dict_attrib['type']
        if datatype == 'string':
            data = str(item.text)
        if datatype == 'int':
                data = int(item.text)
        if datatype == 'logical':
            if 'T' in item.text:
                data = True
            elif 'F' in item.text:
                data = False
    else:
        try:
            data = float(item.text)
        except:
            data = item.text

    return {keyname: data}


def parse_vector(vector):
    """
    Parse vector leaf in vasprun.xml.

    :params:
        vector - vector node from vasprun.xml

    :return:
        dict - dictionary representation of the node.
    """
    dict_attrib = vector.attrib
    data = np.array(vector.text.split())
    try:
        data = data.astype('float')
    except:
        pass

    if vector.tag == 'time':
        keyname = vector.tag + '_' + dict_attrib['name']
    elif 'name' in dict_attrib:
        keyname = dict_attrib['name']
    else:
        keyname = None

    if 'type' in dict_attrib:
        datatype = dict_attrib['type']
        if datatype == 'int':
            data = data.astype('int')

    return {keyname: data}


def vector_generator(vector_list):
    """
    Generator for varray node.

    :params:
        vector - list of vector nodes in varray.

    :return:
        dict - dictionary representation of the vector nodes.
    """
    for vector in vector_list:
        dict_attrib = vector.attrib
        data = np.array(vector.text.split())
        data = data.astype('float')
        if 'name' in dict_attrib:
            keyname = dict_attrib['name']
        else:
            keyname = None

        if 'type' in dict_attrib:
            datatype = dict_attrib['type']
            if datatype == 'int':
                data = data.astype('int')

        yield (keyname, data)


def parse_eigenvectors(node):
    """
    Parse eigenvector node from vasprun.xml.

    :params:
        node - eigenvector node

    :return:
        dict - Dictionary representation of the node.
    """
    arr = node.find('array')
    fields = [item.text for item in arr.findall('field')]
    set1 = arr.find('set')
    set2_list = set1.findall('set')

    dict3 = {}
    dict2 = {}
    for set2 in set2_list:
        set3_list = set2.findall('set')
        key2 = set2.attrib['comment']
        for set3 in set3_list:
            mat = [item.text.split() for item in set3]
            mat = np.array(mat).astype('float').T
            key3 = set3.attrib['comment']
            dict3[key3] = {key: value for (key, value)
                           in zip(fields, mat)}
        dict2[key2] = dict3

    return {node.tag: dict2}


def parse_generic(node):
    """
    Parse generic node in vasprun.xml.

    :params:
        node - node to be parsed.

    :return:
        dict - Dictionary representation of the node.
    """
    if 'name' in node.attrib:
        keyname = node.attrib['name']
    elif 'param' in node.attrib:
        keyname = node.attrib['param']
    else:
        keyname = node.tag
    data = parse_node(node)
    return {keyname: data}


def parse_array(node):
    """
    Parse array node in vasprun.xml.

    :params:
        node - node to be parsed.

    :return:
        dict - Dictionary representation of the node.
    """
    if 'name' in node.attrib:
        keyname = node.attrib['name']
    else:
        keyname = node.tag
    dim = {'dimensions': {'dim_' + item.attrib['dim']: item.text
                          for item in node.findall('dimension')}}
    types = {'types': [item.attrib['type'] if len(item.attrib) > 0
                       else 'float' for item in node.findall('field')]}
    fields = {'fields': [item.text for item in node.findall('field')]}

    vset = node.find('set')
    array_data = parse_set(vset, fields, types)
    data = dict(dim.items() + array_data.items())

    return {keyname: data}


def parse_set(node, fields, types):
    """
    Parse set node in vasprun.xml.

    :params:
        node - node to be parsed.

    :return:
        dict - Dictionary representation of the node.
    """
    var = np.array([[c.text for c in rc] for rc in node]).T
    return {'data': {key: value.astype(vtype) for (key, value, vtype)
            in zip(fields['fields'], var, types['types'])}}


def parse_atoms(node):
    """
    Parse atoms node in vasprun.xml.

    :params:
        node - node to be parsed.

    :return:
        dict - Dictionary representation of the node.
    """
    return {'natoms': int(node.text)}


def parse_types(node):
    """
    Parse types node in vasprun.xml.

    :params:
        node - node to be parsed.

    :return:
        dict - Dictionary representation of the node.
    """
    return {'ntypes': int(node.text)}


def parse_generation(generation):
    """
    Parse generation node in vasprun.xml.

    :params:
        generation - node to be parsed.

    :return:
        dict - Dictionary representation of the node.
    """
    keyname = generation.attrib['param']
    for node in generation:
        data = parse_node(node)

    return {keyname: data}


def parse_varray(varray):
    """
    Parse varray node in vasprun.xml.

    :params:
        varray - node to be parsed.

    :return:
        dict - Dictionary representation of the node.
    """
    keyname = varray.attrib['name']
    vec_list = [item for item in varray if item.tag == 'v']
    vecgen = vector_generator(vec_list)
    data = np.array([value for (key, value) in vecgen])

    return {keyname: data}


def parse_separator(separator):
    """
    Parse generation node in vasprun.xml.

    :params:
        separator - node to be parsed.

    :return:
        dict - Dictionary representation of the node.
    """
    keyname = separator.attrib['name']
    data = parse_node(separator)

    return {keyname: data}

if __name__ == '__main__':
    vasprun = VasprunParser()
    vasprun.parse_vasprun()
    vasprun.to_h5(filename='vasprun.h5')
