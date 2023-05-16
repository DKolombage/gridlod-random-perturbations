import numpy as np
from gridlod import util

def build_randomcheckerboard(Nepsilon, NFine, alpha, beta, p):
    # builds a random checkerboard coefficient with spectral bounds alpha and beta,
    # piece-wise constant on mesh with Nepsilon blocks
    # returns a fine coefficient on mesh with NFine blocks
    '''Di: 
        alpha - background const.
        beta - defect const.
        p - probability = const.
        pList - 1-D array of the probabilities p.
        Nepsilon - Number of "epsilon-blocks" on τ_ε mesh in each direction (1-D array: [x_ε, y_ε, z_ε] if 3D etc.)
        NFine - Number of "fine-blocks" on τ_h mesh in each direction (1-D array: [x_h, y_h, z_h])
        Ntepsilon - Number of ε-blocks on the whole τ_ε mesh = const. (in array format)
        c = determines if there exists a defect(if 1) or not (if 0) on each ε-block. = 1-D array of Ntepsilon elements with values 1 or 0.
        values = Replaces 1 or 0 values from c with the defect coefficients beta or alpha resp. = (1-D array)

        util.tCoordinates - Takes the number of blocks per dimension and return the t-coordinate of the each block. 
                            t ---> name of the element, tCoordinate ---> returns the midpoint of the element t, pCoordinate ---> returns the vertices of the element t
    '''
    Ntepsilon = np.prod(Nepsilon)
    c = np.random.binomial(1,p,Ntepsilon) # Di: (1, p, Ntepsilon) = (# of trials/flips per each coin, probability of having a 1, # of performed tests (one test for each block)) 
    values = alpha + (beta-alpha) * c

    def randomcheckerboard(x):
        '''Di:
            x - the position of any point with its global coordinates on the domain D represented with coordinates in each dimension as an array. 
                x can be an array of elements
        '''
        index = (x*Nepsilon).astype(int)  # Di: position coordinates of x in term of ε-blocks--> gives the lower integer bounds of the coordinates if x is not on the τ_ε grid.
        d = np.shape(index)[1]            # Di: d=1 --> 1-D case, d=2 --> 2-D case etc.

        if d == 1:
            flatindex = index[:]
        if d == 2:
            flatindex = index[:,1]*Nepsilon[0]+index[:,0]   # Di: e.g. An element in the index position index[ , :] = x[x1, y1] = [5, 8] = 85 in the flatindex ---> The considered ε-block is on the 8th row 5th column position  
        if d == 3:
            flatindex = index[:,2]*(Nepsilon[0]*Nepsilon[1]) + index[:,1]*Nepsilon[0] + index[:,0]
        else:
            NotImplementedError('other dimensions not available')

        return values[flatindex]

    xFine = util.tCoordinates(NFine) # Di:  XFine = Returns the tCoordinate (midpoint of each block) of the fine-blocks in each direction.

    return randomcheckerboard(xFine).flatten()  # Di: Gets the alpha, beta values on xFine blocks as a 1-D array (flatten the d-dimensional system into 1-d array)

def build_checkerboardbasis(NPatch, NepsilonElement, NFineElement, alpha, beta):
    # builds a list of coeficients to combine any checkerboard coefficient
    # input: NPatch is number of coarse elements, NepsilonElement and NFineElement the number of cells (per dimension)
    # per coarse element for the epsilon and the fine mesh, respectively; alpha and beta are the spectral bounds of the coefficient
    '''Di:
        NPatch - The number of coarse elements H per dimension inside the sub-domain (Patch) we are interested in.
        NepsilonElement - The number of ε-blocks per dimension per coarse element
        NFineElement - The number of fine-blocks per dimension per coarse element
        Ntepsilon - Number of ε-blocks on the wholse τ_ε mesh = const. (in single element array format: [const.])
        NtFine - Number of fine-blocks on the wholse τ_h mesh = const. (in single element array format)
    '''

    Nepsilon = NPatch * NepsilonElement  # Di: [x_ε, y_ε, z_ε] = [x_H, y_H, z_H]*[x_nε/H, y_nε/H, z_nε/H]
    Ntepsilon = np.prod(Nepsilon)
    NFine = NPatch*NFineElement
    NtFine = np.prod(NFine)

    def checkerboardI(ii):
        '''Di: 
                Idea: To find which element on the Fine grid corresponds to the block ii.
                convertpLinearIndexToCoordIndex(Nepsilon-1, ii)[:] ---> Convert a 1-D index into a DD-coordinate index 
                util.extractElementFine(Nepsilon, NFineElement//NepsilonElement, elementIndex) -----> ???
        '''
        coeff = alpha * np.ones(NtFine) # Di: 1-D array of size NtFine of alpha values. [α,	α, ..., 	α] ---> α values on all fine elements
        #find out which indices on fine grid correspond to element ii on epsilon grid
        elementIndex = util.convertpLinearIndexToCoordIndex(Nepsilon-1, ii)[:] # Di: The corresponding ε-coordinates of element ii wrt fine-coordinates, i.e. Find ij index from ii index
        indices = util.extractElementFine(Nepsilon, NFineElement//NepsilonElement, elementIndex) # Di: We are done if the ε-grid is as same as the fine grid. Else, extract Fine elements from coarse elements: NFineElement//NepsilonElement = floor integer of number of fine-blocks per dimension per coarse element (In this case, The number of ε-blocks=coarse element per dimension
        coeff[indices] = beta # Di: Replace ...??
        return coeff

    checkerboardbasis = list(map(checkerboardI, range(Ntepsilon))) #Di: coeff value (alpha or beta) for each ε-block on the whole domain: A_1, ...A_(Ntε)
    checkerboardbasis.append(alpha*np.ones(NtFine)) # The last element represents A_0 on the note: Defect-less case

    return checkerboardbasis

def build_inclusions_defect_2d(NFine, Nepsilon, bg, val, incl_bl, incl_tr, p_defect, def_val=None):
    # builds a fine coefficient which is periodic with periodicity length 1/epsilon.
    # On the unit cell, the coefficient takes the value val inside a rectangle described by  incl_bl (bottom left) and             
    # incl_tr (top right), otherwise the value is bg
    # with a probability of p_defect the inclusion 'vanishes', i.e. the value is set to def_val (default: bg)
    '''Di: incl_bl (bottom left) etc---> nodeal values?
            bg - background coefficient (The whites)
            val - inclusion value (The blacks) ---> small black blocks on relatively bigger white blocks ---> position can be chosen freely by defining incl_bl & incl_tr
            incl_bl -inclusion bottom left vertex value of the black block
            incl_tr - inclusion top right vertex value of the black block
            p_defect - probability of defect,  a defect means now, we erase the inclusion (the black box on the white block)
            def_val -  This is to indicate if the black box is now grey, meaning, we don't entirely erase the black block to get a defect but its colour intensity reduces.
    '''

    assert(np.all(incl_bl) >= 0.)
    assert(np.all(incl_tr) <= 1.)
    assert(p_defect < 1.)

    if def_val is None:
        def_val = bg

    #probability of defect is p_defect
    c = np.random.binomial(1, p_defect, np.prod(Nepsilon))   

    aBaseSquare = bg*np.ones(NFine) # Di: bg*([1, 1])
    flatidx = 0  
    for ii in range(Nepsilon[0]):          # Di: Iterate through ε-blocks on the x-axis of the rectangle
        for jj in range(Nepsilon[1]):      # Di: Iterate through ε-blocks on the y-axis of the rectangle
            startindexcols = int((ii + incl_bl[0]) * (NFine/Nepsilon)[0]) # ()* Number of fine-blocks per ε-block
            stopindexcols = int((ii + incl_tr[0]) * (NFine/Nepsilon)[0])
            startindexrows = int((jj + incl_bl[1]) * (NFine/Nepsilon)[1])
            stopindexrows = int((jj + incl_tr[1]) * (NFine/Nepsilon)[1])
            if c[flatidx] == 0:
                aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = val
            else:
                aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = def_val
            flatidx += 1

    return aBaseSquare.flatten()  # Di: 1-D array of aBaseSquare values

def build_inclusions_change_2d(NFine, Nepsilon, bg, val, incl_bl, incl_tr, p_defect, model):
    # builds a fine coefficient which is periodic with periodicity length 1/epsilon.
    # On the unit cell, the coefficient takes the value val inside a rectangle described by  incl_bl (bottom left) and
    # incl_tr (top right), otherwise the value is bg
    # with a probability of p_defect the inclusion 'changes', where three models are implemented:
    #    -filling the whole scaled unit cell (fill)
    #    -shifting the inclusion to def_bl, def_br
    #    - L-shape, i.e. erasing only the area def_bl to def_br

    assert(np.all(incl_bl) >= 0.)
    assert(np.all(incl_tr) <= 1.)
    assert(p_defect < 1.)

    assert(model['name'] in ['inclfill', 'inclshift', 'inclLshape'])

    #probability of defect is p_defect
    c = np.random.binomial(1, p_defect, np.prod(Nepsilon))

    aBaseSquare = bg*np.ones(NFine)
    flatidx = 0
    for ii in range(Nepsilon[0]):
        for jj in range(Nepsilon[1]):
            startindexcols = int((ii + incl_bl[0]) * (NFine/Nepsilon)[0])
            stopindexcols = int((ii + incl_tr[0]) * (NFine/Nepsilon)[0])
            startindexrows = int((jj + incl_bl[1]) * (NFine/Nepsilon)[1])
            stopindexrows = int((jj + incl_tr[1]) * (NFine/Nepsilon)[1])
            if c[flatidx] == 0: # no defect
                aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = val
            else:
                if model['name'] == 'inclfill':
                    startdefindexcols = int((ii) * (NFine / Nepsilon)[0])
                    stopdefindexcols = int((ii + 1) * (NFine / Nepsilon)[0])
                    startdefindexrows = int((jj) * (NFine / Nepsilon)[1])
                    stopdefindexrows = int((jj + 1) * (NFine / Nepsilon)[1])
                    aBaseSquare[startdefindexrows:stopdefindexrows, startdefindexcols:stopdefindexcols] = val
                if model['name'] == 'inclshift':
                    def_bl = model['def_bl']
                    def_tr = model['def_tr']
                    startdefindexcols = int((ii + def_bl[0]) * (NFine / Nepsilon)[0])
                    stopdefindexcols = int((ii + def_tr[0]) * (NFine / Nepsilon)[0])
                    startdefindexrows = int((jj + def_bl[1]) * (NFine / Nepsilon)[1])
                    stopdefindexrows = int((jj + def_tr[1]) * (NFine / Nepsilon)[1])
                    aBaseSquare[startdefindexrows:stopdefindexrows, startdefindexcols:stopdefindexcols] = val
                if model['name'] == 'inclLshape':
                    #first, put a normal inclusion
                    aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = val
                    # erase now the complement of the Lshape in the inclusion
                    def_bl = model['def_bl']
                    def_tr = model['def_tr']
                    startdefindexcols = int((ii + def_bl[0]) * (NFine / Nepsilon)[0])
                    stopdefindexcols = int((ii + def_tr[0]) * (NFine / Nepsilon)[0])
                    startdefindexrows = int((jj + def_bl[1]) * (NFine / Nepsilon)[1])
                    stopdefindexrows = int((jj + def_tr[1]) * (NFine / Nepsilon)[1])
                    aBaseSquare[startdefindexrows:stopdefindexrows, startdefindexcols:stopdefindexcols] = bg
            flatidx += 1

    return aBaseSquare.flatten()

def build_inclusionbasis_2d(NPatch, NEpsilonElement, NFineElement, bg, val, incl_bl, incl_tr, defval=None):
    Nepsilon = NPatch * NEpsilonElement
    NFine = NPatch * NFineElement
    if defval is None:
        defval = bg

    assert (np.all(incl_bl) >= 0.)
    assert (np.all(incl_tr) <= 1.)

    aBaseSquare = bg * np.ones(NFine)
    for ii in range(Nepsilon[0]):
        for jj in range(Nepsilon[1]):
            startindexcols = int((ii + incl_bl[0]) * (NFine / Nepsilon)[0])
            stopindexcols = int((ii + incl_tr[0]) * (NFine / Nepsilon)[0])
            startindexrows = int((jj + incl_bl[1]) * (NFine / Nepsilon)[1])
            stopindexrows = int((jj + incl_tr[1]) * (NFine / Nepsilon)[1])
            aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = val

    #aBase = aBaseSquare.flatten()

    def inclusion_defectI(ii):
        aSquare = np.copy(aBaseSquare)
        tmp_indx = np.array([ii % Nepsilon[1], ii // Nepsilon[1]])
        startindexcols = int((tmp_indx[0] + incl_bl[0]) * (NFine / Nepsilon)[0])
        stopindexcols = int((tmp_indx[0] + incl_tr[0]) * (NFine / Nepsilon)[0])
        startindexrows = int((tmp_indx[1] + incl_bl[1]) * (NFine / Nepsilon)[1])
        stopindexrows = int((tmp_indx[1] + incl_tr[1]) * (NFine / Nepsilon)[1])
        aSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = defval
        return aSquare.flatten()

    coeffList = list(map(inclusion_defectI, range(np.prod(Nepsilon))))
    coeffList.append(aBaseSquare.flatten())

    return coeffList

def build_inclusionbasis_change_2d(NPatch, NEpsilonElement, NFineElement, bg, val, incl_bl, incl_tr, model):
    Nepsilon = NPatch * NEpsilonElement
    NFine = NPatch * NFineElement

    assert (np.all(incl_bl) >= 0.)
    assert (np.all(incl_tr) <= 1.)
    assert(model['name'] in ['inclfill', 'inclshift', 'inclLshape'])

    aBaseSquare = bg * np.ones(NFine)
    for ii in range(Nepsilon[0]):
        for jj in range(Nepsilon[1]):
            startindexcols = int((ii + incl_bl[0]) * (NFine / Nepsilon)[0])
            stopindexcols = int((ii + incl_tr[0]) * (NFine / Nepsilon)[0])
            startindexrows = int((jj + incl_bl[1]) * (NFine / Nepsilon)[1])
            stopindexrows = int((jj + incl_tr[1]) * (NFine / Nepsilon)[1])
            aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = val

    #aBase = aBaseSquare.flatten()

    def inclusion_defectI(ii):
        aSquare = np.copy(aBaseSquare)
        tmp_indx = np.array([ii % Nepsilon[1], ii // Nepsilon[1]])
        if model['name'] == 'inclfill':
            startdefindexcols = int((tmp_indx[0]) * (NFine / Nepsilon)[0])
            stopdefindexcols = int((tmp_indx[0] + 1) * (NFine / Nepsilon)[0])
            startdefindexrows = int((tmp_indx[1]) * (NFine / Nepsilon)[1])
            stopdefindexrows = int((tmp_indx[1] + 1) * (NFine / Nepsilon)[1])
            aSquare[startdefindexrows:stopdefindexrows, startdefindexcols:stopdefindexcols] = val
        if model['name'] == 'inclshift':
            def_bl = model['def_bl']
            def_tr = model['def_tr']
            #first erase the inclusion
            startindexcols = int((tmp_indx[0] + incl_bl[0]) * (NFine / Nepsilon)[0])
            stopindexcols = int((tmp_indx[0] + incl_tr[0]) * (NFine / Nepsilon)[0])
            startindexrows = int((tmp_indx[1] + incl_bl[1]) * (NFine / Nepsilon)[1])
            stopindexrows = int((tmp_indx[1] + incl_tr[1]) * (NFine / Nepsilon)[1])
            aSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = bg
            #now put the inclusion at the new place
            startdefindexcols = int((tmp_indx[0] + def_bl[0]) * (NFine / Nepsilon)[0])
            stopdefindexcols = int((tmp_indx[0] + def_tr[0]) * (NFine / Nepsilon)[0])
            startdefindexrows = int((tmp_indx[1] + def_bl[1]) * (NFine / Nepsilon)[1])
            stopdefindexrows = int((tmp_indx[1] + def_tr[1]) * (NFine / Nepsilon)[1])
            aSquare[startdefindexrows:stopdefindexrows, startdefindexcols:stopdefindexcols] = val
        if model['name'] == 'inclLshape':  # erase the complement of the Lshape in the inclusion
            def_bl = model['def_bl']
            def_tr = model['def_tr']
            startdefindexcols = int((tmp_indx[0] + def_bl[0]) * (NFine / Nepsilon)[0])
            stopdefindexcols = int((tmp_indx[0] + def_tr[0]) * (NFine / Nepsilon)[0])
            startdefindexrows = int((tmp_indx[1] + def_bl[1]) * (NFine / Nepsilon)[1])
            stopdefindexrows = int((tmp_indx[1] + def_tr[1]) * (NFine / Nepsilon)[1])
            aSquare[startdefindexrows:stopdefindexrows, startdefindexcols:stopdefindexcols] = bg
        return aSquare.flatten()

    coeffList = list(map(inclusion_defectI, range(np.prod(Nepsilon))))
    coeffList.append(aBaseSquare.flatten())

    return coeffList