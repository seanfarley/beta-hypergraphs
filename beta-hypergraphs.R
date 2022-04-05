################################################################################
# Given a file of co-verticeships of the following format:
# vertex1,vertex2,vertex3
# vertex4
# vertex5,vertex6
# the function returns a list of vertices, and a list of hyperedges for each entry in filename.
# grep can be used from the terminal to easily preprocess a bib file to bring the article
# data to the required format.
################################################################################
Read.Hypergraph.From.File<-function(filename, simple = TRUE){
	H.file  = file(filename);    
	on.exit(close(H.file))
	hyperedges.str  = readLines(H.file);
	hyperedges = list()
	vertices = c();
	for (e in hyperedges.str){
		v.list = strsplit(e,split=',');
		# sort the hyperedges so that we can recognize multiple edges
		v.ordered.list = list(sort(unlist(v.list)))
		hyperedges = append(hyperedges, v.ordered.list)
		for (vertex in v.list){
			vertices = c(vertices, vertex)
		}
	}
	vertices = unique(vertices);
	if (simple){
		hyperedges = unique (hyperedges)	
	}
	return (list(vertices,hyperedges))
}

Get.Induced.Subgraph<-function(vertices, hyperedges, k){
	inducedH = list()
	for(e in hyperedges){
		if (all(e %in% vertices)){
				inducedH=append(inducedH,list(e))
		}
	}
	return(inducedH)
}

Get.Size.k.Hyperedges<-function(hyperedges, k){
# Given a list of hyperedges and an integer k returns the subset of the hyperedges
# that have size k.
	k.edges=list()
	if (is.list(k)){
		for (e in hyperedges){
			if (length(e) %in% k){
				k.edges = append(k.edges,list(e))
			}
		}	
	}
	else {
		for (e in hyperedges){
			if (length(e)==k){
				k.edges = append(k.edges,list(e))
			}
		}		
	}
	return(k.edges)
}

Get.Degree.Sequence<-function(hyperedges, vertices = NULL){
	if (is.null(vertices)){
		vertices = unique(unlist(hyperedges))	
	}
	n=length(vertices)
	d = rep(0,n)
	for (i in 1:n){ 
		for (e in hyperedges){
			if(vertices[i] %in% e){d[i]=d[i]+1}		
		}
	}
	return (d)
}

Get.Dense.Subgraph<-function(hyperedges, d.threshold=1, num.low.degree.vertices =0){
	d = Get.Degree.Sequence(hyperedges)
	vertices = unique(unlist(hyperedges))
	# Recursively remove all low degree vertices 
	# and edges that contain them from graph until the graph 
	# contains no low degree vertices or is empty
	while ( (length(vertices)>0) && (length(vertices[d<=d.threshold])>num.low.degree.vertices) ){
		vertices = vertices[d>1]
		hyperedges = Get.Induced.Subgraph(vertices, hyperedges)
		d = Get.Degree.Sequence(hyperedges)
	}
#	print("under construction")
	return(hyperedges)
}

Get.Smallest.Subgraph.Containing.Vertices<-function(vertices, hyperedges, k=NULL){
	subgraph = list()
	if (!is.null(k)){
		hyperedges = Get.Size.k.Hyperedges(hyperedges,k)
	}
	for(e in hyperedges){
		if (any(e %in% vertices)){
				subgraph=append(subgraph,list(e))
		}
	}
	return(subgraph)
}

Get.Smallest.Induced.Subgraph.Containing.Vertices<-function(vertices, hyperedges, k=NULL){
	new.vertex.set = list()
	if (!is.null(k)){
		hyperedges = Get.Size.k.Hyperedges(hyperedges,k)
	}
	for(e in hyperedges){
		if (any(e %in% vertices)){
				new.vertex.set=append(new.vertex.set,e)
		}
	}
	new.vertex.set = unique(new.vertex.set)
	return(Get.Induced.Subgraph(new.vertex.set, hyperedges,k=NULL))
}

# Use a fixed point algorithm to get the MLE
Estimate.Beta.Fixed.Point<-function(degrees, case="k-uniform", k, max.iter=500, tol=0.0001, beta = NULL){
	n=length(degrees)
	if (is.null(beta)){
		beta=rep(0,n)#log(degrees)#rep(0,n) #not sure what a good starting point would be
	}
	convergence=FALSE
	diff.list = list()
	steps=0
	#### There are more efficient methods for calculating e^{beta_S} for each S in n\choose k-1, e.g using a dynamic algorithm
	#sets = combinations(n,k-1,repeats.allowed=FALSE) #old code, defunct package?
	sets = t(combn(n,k-1))
	prod.exp.beta=rep(1,nrow(sets))
	while(convergence==FALSE && steps < max.iter){
		exp.beta=exp(beta) #calculate e^beta
		old.beta = beta
		if(any(is.infinite(old.beta))){
#			stop("Infinite beta estimate.")			
			return(NULL)
		}
		for (i in 1:nrow(sets) ){
			tuple=sets[i,] # tuple is a (k-1)-tuple of []1,2,...n]
#			print(tuple)
#			print(exp.beta[tuple])
			prod.exp.beta[i] = prod(exp.beta[tuple]) #calculate the product of e^beta for 
			if (is.infinite(prod.exp.beta[i])){
				print("Infinite beta.")
				return(NULL)
				}
		}
		# Calculate the beta estimate for each beta[i]
		##### currently very slow #####
		for (i in 1:n){
			sum.q.beta=0
			for (j in 1:nrow(sets)){
				tuple=sets[j,] # tuple is a (k-1)-tuple of [1,2,...n]	
				if (!(i %in% tuple)){
					sum.q.beta = sum.q.beta + prod.exp.beta[j]/(1+(prod.exp.beta[j]*exp.beta[i]))
					if (is.infinite(sum.q.beta)){
						print("Infinite beta.")
						return(NULL)
						}
				}	
				beta[i]=log(degrees[i])-log(sum.q.beta)
			}
		}
		diff = max(abs(old.beta-beta))
		print(paste("diff=", diff,  "-------- steps=",steps))
		print(beta)
		if (diff < tol){
			convergence=TRUE
		}
		steps=steps+1;
	}
	print(steps)
	print(diff)
	if (diff %in% diff.list){
		print("Divergent Subsequence Detected.")
		return(NULL)
		}
	diff.list = append(diff.list, list(diff))
	
	if (steps == max.iter){
		return(NULL)
		}
	else
		return (beta)
}
Estimate.Beta.Fixed.Point.General.Case<-function(degrees, k.list, max.iter=500, tol=1e-04, beta = NULL){
# Use a fixed point algorithm to get the MLE
# Accepts a list of edge-sizes 	
	n = length(degrees)
	if (is.null(beta)){ beta=rep(0,n)#log(degrees)#rep(0,n) #pick a starting point 
		}
	convergence=FALSE
	steps=0
	#### There are more efficient methods for calculating e^{beta_S} for each S in n\choose k-1, e.g using a dynamic algorithm
	#sets = combinations(n,k-1,repeats.allowed=FALSE) #old code, defunct package?
	all.index.sets = list()
	index.k = 0
	prod.exp.beta.list = list()
	for (k in k.list){
		print(k)
		index.k = index.k+1
		sets = t(combn(n,k-1))
		all.index.sets = append(all.index.sets, list(sets))
		prod.exp.beta.list=append(prod.exp.beta.list, list(rep(1, nrow(sets))))
	}
	
	diff.list = list()
	
	while(convergence==FALSE && steps < max.iter){
		exp.beta=exp(beta)
		old.beta = beta # To estimate convergence 
		if (any(is.infinite(old.beta))){
#			stop("Infinite beta estimate.")	
			print("Infinite beta.")
			return(NULL)
		}
		# Calculate e^{b_s} for all possible (k-1)-tuples
		for (index.k in 1:length(k.list)) {
			sets = all.index.sets[[index.k]]
			prod.exp.beta = prod.exp.beta.list[[index.k]]
			for (t in 1:nrow(sets) ){
				tuple=sets[t,] # tuple is a (k-1)-tuple of [1,2,...n] for some k in k.list
				prod.exp.beta[t] = prod(exp.beta[tuple])
				if (is.infinite(prod.exp.beta[t])){
					print("Infinite beta.")
					return(NULL)
				}
			}
			prod.exp.beta.list[[index.k]] = prod.exp.beta
#			print(paste("k = ", k.list[index.k], " complete.")) #testing
#			print(prod(prod.exp.beta)) #testing
		}
		# Calculate the beta estimate for each beta[i]
		for (i in 1:n) {
			sum.q.beta= rep(0, length(k.list))
			for (index.k in 1:length(k.list)){
				sets = all.index.sets[[index.k]]
				prod.exp.beta = prod.exp.beta.list[[index.k]]
				for (j in 1:nrow(sets)) {
					tuple=sets[j,] # tuple is a (k-1)-tuple of [1,2,...n]	for each k in k.list
					# See alternative code fragment in Estimate.Degree.Sequence.From.Beta and decide which is faster 
					if (!(i %in% tuple)) {
						sum.q.beta[index.k] = sum.q.beta[index.k] + prod.exp.beta[j]/(1+(prod.exp.beta[j]*exp.beta[i]))
						if (is.infinite(sum.q.beta[index.k])){
							print("Infinite beta.")
							return(NULL)
						}
					}					
				}
			}
			beta[i] = log(degrees[i]) - log(sum(sum.q.beta))
		}
		diff = max( abs(old.beta-beta) )
		print(paste("diff=", diff,  "-------- steps=",steps))
		print(beta)
		if (diff < tol){ convergence=TRUE }
		if (diff %in% diff.list){
			print("Divergent Subsequence Detected.")
			return(NULL)
			}
		diff.list = append(diff.list, list(diff))
		steps = steps+1;
	}
	print(steps)
	print(diff)
	if (steps == max.iter){return(NULL)}
	else return (beta)
}

Estimate.Degree.Sequence.From.Beta<-function(beta, k.list){
	#For testing purposes, layered k-uniform model
	n=length(beta)
	d = rep(0, n)
	exp.beta=exp(beta)
	for (i in 1:n){
		for (k in k.list){
			if(i==1){ tuples = combn(2:n,k-1) }
			else if(i==n){ tuples = combn(n-1,k-1) }
			else { tuples = combn(c(1:(i-1),(i+1):n),k-1) }
#alternative
#			apply(combn(n,k-1), 2, function(x) ifelse(i%in%x, beta.s=NULL, beta.s=beta[x]))			
			for (j in 1:ncol(tuples)){
				tuple = tuples[,j]
				exp.beta.tuple = exp(sum(beta[tuple]))
				d[i] = d[i] + (exp.beta.tuple*exp.beta[i]) / ( 1 + ( exp.beta.tuple*exp.beta[i] ) )
			}
		}
	}
	return(d)
}

Construct.IPS.Table<-function(vertices, hyperedges, model="k-uniform", k=2){
   print("UNDER CONSTRUCTION")
	n=length(vertices);
	t=array(0, dim=rep(n,k))	#initialize r-way array
#	t=rep(1/n,n^k);			#initialize r-way array
#	dim(t)<-c(rep(n,k)); 
	for (i in 1:n){
		t[rep(i,k)]=0;
		}
	
#	while(i<=3 && j<=4 && k<=5){ 
#		if ( i!=j && j!=l && i!=k){
#			v[i,j,k]=1/n;
#			}
#		}
	return(table)
}

# Construct a graph 
Reduce.To.Graph<-function(hyperedges, vertices=NULL, simple = TRUE){
	if (is.null(vertices)){
		vertices = unique(unlist(hyperedges))
	}
	edges = list()
	for (e in hyperedges){
		if (length(e) == 2){
			edges = append(edges, list(e))
		}
		else{
			pairs = combn(e,2) #returns matrix
			for (i in 1:ncol(pairs)){
				edges = append(edges, list(pairs[,i]))	
			}			
		}
	}
	if (simple){
		edges = unique (edges)	
	}
	return (list(vertices, edges))
}

Get.Probability.Under.k.uniform.Model<-function(beta, d, k, psi.beta = NULL){
	n=length(beta)
	if (is.null(psi.beta)){
		psi.beta=0
		tuples = combn(1:n,k)	
		for (j in 1:ncol(tuples)){
			tuple = tuples[,j]	
			psi.beta = psi.beta +log(1+ exp(sum(beta[tuple])))
		}
	}
	prob = exp(sum(d*beta)-psi.beta)
	return(prob)
}

Get.Probability.Under.Layered.Model<-function(beta.list, d.list, k.list, psi.beta.list = NULL){
	prob=1
	if (is.null(psi.beta.list)){ psi.beta.list=rep(NULL,length(k.list))}
	for (i in 1:length(k.list)){
		prob=prob * Get.Probability.Under.k.uniform.Model(beta.list[[i]], d.list[[i]],k.list[[i]], psi.beta.list[[i]])
	}
	return(prob)
}

Get.Probability.Under.General.Model<-function(beta, d, k.list, psi.beta = NULL){
	n=length(beta)
	if (is.null(psi.beta)){
		psi.beta=0
		for (k in k.list){
			tuples = combn(1:n,k)	
			for (j in 1:ncol(tuples)){
				tuple = tuples[,j]	
				psi.beta = psi.beta +log(1+ exp(sum(beta[tuple])))
			}
		}
	}
	prob = exp(sum(d*beta)-psi.beta)
	return(prob)
}

Get.Probability.Of.Edge.Under.Model<-function(beta, edge){
	exp.edge =exp(sum(beta[edge]))
	prob.edge = exp.edge/(1+exp.edge)
	return(prob.edge)
}

Draw.Random.Hypergraph.From.Model<-function(beta, k){
	hyperedges = list()
	n=length(beta)
	tuples = combn(1:n,k)
	for (i in 1:ncol(tuples)){
		tuple = tuples[,i]
		exp.sum.betas=exp(sum(beta[tuple]))
		prob.edge=(exp.sum.betas)/(1+ exp.sum.betas)
		coin = sample(c(TRUE,FALSE), size=1,prob=c(prob.edge,1-prob.edge))
		if (coin){hyperedges = append(hyperedges,list(tuple))}
	}
	return(hyperedges)
}

