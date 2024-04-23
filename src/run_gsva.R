rm(list=ls())
set.seed(46)


get_ssgsea<-function(inputf, resultf, msigdbf){
  library("GSVA")
  library("data.table")
  library("GSEABase")
  rgz <- gzfile(inputf,"r")
  exp_mat <- read.csv(rgz, row.names=1, header=TRUE)
  close(rgz)
  print(exp_mat[1:5,1:5])
  if(max(exp_mat) > 50){
    exp_mat <- log2(exp_mat + 1)
  }
  gmt <- getGmt(gzfile(msigdbf), geneIdType= SymbolIdentifier(), sep="\t")
  geneSets <- geneIds(gmt)
  filter_gmt <- filterGeneSets(geneSets, min.sz=5, max.sz=300)
  expmat <- as.matrix(exp_mat)
  crc_es <- gsva(expmat, filter_gmt, mx.diff=TRUE, verbose=TRUE, parallel.sz=8, method="ssgsea")
  wgz <- gzfile(resultf,"w")
  write.csv(crc_es,wgz)
  close(wgz)
}


library("optparse")
option_list <- list(make_option(c("-i","--input.file"), type = "character",help="the exp matrix"))
opt <- parse_args(OptionParser(option_list=option_list))

if (is.null(opt$input.file)){
  print(opt$help)
}


if (file.exists(paste0(getwd(), "/features/msigdb.v7.4.symbols.gmt.gz"))){
  msigdbf <- paste0(getwd(), "/features/msigdb.v7.4.symbols.gmt.gz")
}else if (file.exists("../features/msigdb.v7.4.symbols.gmt.gz")){
  msigdbf <- "../features/msigdb.v7.4.symbols.gmt.gz"
}else{
  stop("check your msigdb in features directory")
}


resultf <- gsub(".csv.gz", "_ES.csv.gz", opt$input.file)
print(paste0("result will saved in: ", resultf))
get_ssgsea(opt$input.file, resultf, msigdbf)


