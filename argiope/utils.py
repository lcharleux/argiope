import pickle
import gzip
import copy
import subprocess
import argiope
import os
import inspect

MODPATH = os.path.dirname(inspect.getfile(argiope))
################################################################################
# PICKLE/GZIP RELATED
################################################################################


def load(path):
    """
    Loads a file.
    """
    return pickle.load(gzip.open(path))

################################################################################
# META CLASSES
################################################################################


class Container:
    """
    A container meta class with utilities
    """

    def save(self, path):
        """
        Saves the instance into a compressed serialized file.
        """
        pickle.dump(self, gzip.open(path, "w"), 3)

    def copy(self):
        """
        Returns a copy of self.
        """
        return copy.deepcopy(self)


################################################################################
# SUBPROCESS RELATED
################################################################################
def run_gmsh(gmsh_path="gmsh", gmsh_space=3, gmsh_options="",
             name="dummy.geo", workdir="./"):
    p = subprocess.Popen("{0} -{1} {2} {3}".format(gmsh_path, gmsh_space,
                                                   gmsh_options, name),
                         cwd=workdir,
                         shell=True,
                         stdout=subprocess.PIPE)
    trash = p.communicate()


################################################################################
# MISC
################################################################################


def list_to_string(l=range(200), width=40, indent="  "):
    """
    Converts a list-like to string with given line width.
    """
    l = [str(v) + "," for v in l]
    counter = 0
    out = "" + indent
    for w in l:
        s = len(w)
        if counter + s > width:
            out += "\n" + indent
            counter = 0
        out += w
        counter += s
    return out.strip(",")


def _equation(nodes=(1, 2), dofs=(1, 1), coefficients=(1., 1.),
              comment=None):
    """
    Returns an Abaqus INP formated string for a given linear equation.
    """
    N = len(nodes)
    if comment == None:
        out = ""
    else:
        out = "**EQUATION: {0}\n".format(comment)
    out += "*EQUATION\n  {0}\n  ".format(N)
    out += "\n  ".join([",".join([str(nodes[i]),
                                  str(int(dofs[i])),
                                  str(coefficients[i])]) for i in range(N)])
    return out


def _unsorted_set(df, label, **kwargs):
    """
    Returns a set as inp string with unsorted option.
    """
    out = "*NSET, NSET={0}, UNSORTED\n".format(label)
    labels = df.index.values
    return out + argiope.utils.list_to_string(labels, **kwargs)
