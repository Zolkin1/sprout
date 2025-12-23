
# This is a class to manage multiple skills (i.e. multiple libraries and/or trajectories)

# The idea here is that we can hold a list of other managers (trajectory and/or library), referred to as sub-managers
#   then when we access something in this manager we also need an env index.
#   Each sub-manager is associated with a set of env indices. So we can access the correct sub-manager
#   and call the sub-managers function.