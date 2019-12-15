#include<cstdlib>
#include<algorithm>
#include<complex>
#include<iostream>
#include<fstream>
#include<iomanip>
#include<map>
#include<utility>
#include<vector>
#include<numeric>
#include <functional>
#if defined(USE_MPI)
#include<mpi.h>
#endif

#include "Configuration.h"
#include "type_traits/container_traits_multi.h"
#include "io/hdf_multi.h"
#include "io/hdf_archive.h"

#include "AFQMC/config.h"
#include "AFQMC/Utilities/Utils.hpp"
#include "AFQMC/Utilities/kp_utilities.hpp"
#include "AFQMC/Hamiltonians/RealDenseHamiltonian.h"
#include "AFQMC/SlaterDeterminantOperations/rotate.hpp"

namespace qmcplusplus
{

namespace afqmc
{

#ifndef QMC_COMPLEX

HamiltonianOperations RealDenseHamiltonian::getHamiltonianOperations(bool pureSD,
	     bool addCoulomb, WALKER_TYPES type, std::vector<PsiT_Matrix>& PsiT,
	     double cutvn, double cutv2, TaskGroup_& TGprop, TaskGroup_& TGwfn,
	     hdf_archive& hdf_restart) {

  using shmIMatrix = boost::multi::array<int,2,shared_allocator<int>>;
  using shmCVector = boost::multi::array<ComplexType,1,shared_allocator<ComplexType>>;
  using shmCMatrix = boost::multi::array<ComplexType,2,shared_allocator<ComplexType>>;
  using shmRMatrix = boost::multi::array<RealType,2,shared_allocator<RealType>>;
  using shmCTensor = boost::multi::array<ComplexType,3,shared_allocator<ComplexType>>;
  using shmSpMatrix = boost::multi::array<SPComplexType,2,shared_allocator<SPComplexType>>;
  using shmSpRMatrix = boost::multi::array<SPRealType,2,shared_allocator<SPRealType>>;
  using shmSp3Tensor = boost::multi::array<SPComplexType,3,shared_allocator<SPComplexType>>;
  using IVector = boost::multi::array<int,1>;
  using CMatrix = boost::multi::array<ComplexType,2>;
  using SpMatrix = boost::multi::array<SPComplexType,2>;
  using SpVector_ref = boost::multi::array_ref<SPComplexType,1>;
  using SpMatrix_ref = boost::multi::array_ref<SPComplexType,2>;
  using SpRMatrix = boost::multi::array<SPRealType,2>;
  using SpRMatrix_ref = boost::multi::array_ref<SPRealType,2>;
  using CMatrix_ref = boost::multi::array_ref<ComplexType,2>;
  using RMatrix_ref = boost::multi::array_ref<RealType,2>;
  using Sp3Tensor_ref = boost::multi::array_ref<SPComplexType,3>;

  if(type==COLLINEAR)
    assert(PsiT.size()%2 == 0);
  int nspins = ((type!=COLLINEAR)?1:2);
  int ndet = PsiT.size()/nspins;
  int nup = PsiT[0].size(0); 
  int ndown = 0; 
  if(nspins == 2) ndown = PsiT[1].size(0); 
  int NEL = nup+ndown; 

  hdf_archive dump(TG.Global());
  // right now only Node.root() reads
  if( TG.Node().root() ) {
    if(!dump.open(fileName,H5F_ACC_RDONLY)) {
      app_error()<<" Error opening integral file in THCHamiltonian. \n";
      APP_ABORT("");
    }
    if(!dump.push("Hamiltonian",false)) {
      app_error()<<" Error in THCHamiltonian::getHamiltonianOperations():"
                 <<" Group not Hamiltonian found. \n";
      APP_ABORT("");
    }
  }

  std::vector<int> Idata(8);
  if( TG.Global().root() ) {
    if(!dump.readEntry(Idata,"dims")) {
      app_error()<<" Error in RealDenseHamiltonian::getHamiltonianOperations():"
                 <<" Problems reading dims. \n";
      APP_ABORT("");
    }
  }
  TG.Global().broadcast_n(Idata.begin(),8,0);

  ValueType E0;
  if( TG.Global().root() ) {
    std::vector<RealType> E_(2);
    if(!dump.readEntry(E_,"Energies")) {
      app_error()<<" Error in RealDenseHamiltonian::getHamiltonianOperations():"
                 <<" Problems reading Energies. \n";
      APP_ABORT("");
    }
    E0 = E_[0]+E_[1];
  }
  TG.Global().broadcast_n(&E0,1,0);

  int global_ncvecs = Idata[7]; 
  int nc0, ncN;
  int node_number = TGprop.getLocalGroupNumber();
  int nnodes_prt_TG = TGprop.getNGroupsPerTG();
  std::tie(nc0,ncN) = FairDivideBoundary(node_number,global_ncvecs,nnodes_prt_TG);
  int local_ncv = ncN-nc0;

  shmRMatrix H1({NMO,NMO},shared_allocator<RealType>{TG.Node()});
  shmSpRMatrix Likn({NMO*NMO,local_ncv}, shared_allocator<SPRealType>{TG.Node()});

  if( TG.Node().root() ) {
    // now read H1, use ref to avoid issues with shared pointers!
    RMatrix_ref h1_(to_address(H1.origin()),H1.extensions());
    if(!dump.readEntry(h1_,std::string("hcore"))) {
      app_error()<<" Error in RealDenseHamiltonian::getHamiltonianOperations():"
               <<" Problems reading /Hamiltonian/hcore. \n";
      APP_ABORT("");
    }
    // read L
    if(!dump.push("DenseFactorized",false)) {
      app_error()<<" Error in RealDenseHamiltonian::getHamiltonianOperations():"
                 <<" Group DenseFactorized not found. \n";
      APP_ABORT("");
    }
    SpRMatrix_ref L(to_address(Likn.origin()),Likn.extensions()); 
    hyperslab_proxy<SpRMatrix_ref,2> hslab(L,std::array<int,2>{NMO*NMO,global_ncvecs},
                                             std::array<int,2>{NMO*NMO,local_ncv},
                                             std::array<int,2>{0,nc0});
    if(!dump.readEntry(hslab,std::string("L"))) {
      app_error()<<" Error in RealDenseHamiltonian::getHamiltonianOperations():"
               <<" Problems reading /Hamiltonian/DenseFactorized/L. \n";
      APP_ABORT("");
    }
    if(Likn.size(0) != NMO*NMO || Likn.size(1) != local_ncv) {
      app_error()<<" Error in RealDenseHamiltonian::getHamiltonianOperations():"
             <<" Problems reading /Hamiltonian/DenseFactorized/L. \n"
             <<" Unexpected dimensins: " <<Likn.size(0) <<" " <<Likn.size(1) <<std::endl;
      APP_ABORT("");
    }
    dump.pop();
  }
  TG.Node().barrier();

  shmCMatrix vn0({NMO,NMO},shared_allocator<ComplexType>{TG.Node()});
  shmCMatrix haj({ndet,NEL*NMO},shared_allocator<ComplexType>{TG.Node()});
  std::vector<shmSp3Tensor> Lank;
  Lank.reserve(PsiT.size()); 
  for(int nd=0; nd<PsiT.size(); nd++) 
    Lank.emplace_back(shmSp3Tensor({PsiT[nd].size(0),local_ncv,NMO},shared_allocator<SPComplexType>{TG.Node()}));
  int nrow = NEL;
  if(ndet > 1) nrow = 0;  // not used if ndet>1
  shmSpMatrix Lakn({nrow*NMO,local_ncv},shared_allocator<SPComplexType>{TG.Node()});
  TG.Node().barrier();

  // for simplicity
  CMatrix H1C({NMO,NMO});
  copy_n_cast(to_address(H1.origin()),NMO*NMO,H1C.origin());

  int nt=0;
  for(int nd=0; nd<ndet; nd++) {
    // haj and add half-transformed right-handed rotation for Q=0
    if( nd%TG.Node().size() != TG.Node().rank() ) continue;  
    if(type==COLLINEAR) {
      CMatrix_ref haj_r(to_address(haj[nd].origin()),{nup,NMO});
      ma::product(PsiT[2*nd],H1C,haj_r);
      CMatrix_ref hbj_r(to_address(haj[nd].origin())+(nup*NMO),{ndown,NMO});
      if(ndown>0) ma::product(PsiT[2*nd+1],H1C,hbj_r);
    } else {
      CMatrix_ref haj_r(to_address(haj[nd].origin()),{nup,NMO});
      ma::product(ComplexType(2.0),PsiT[nd],H1C,ComplexType(0.0),haj_r);
    }
  }
  // Generate Lank 
  // distribute work over equivalent nodes in TGprop.TG() accross TG.Global()  
  auto Qcomm(TG.Global().split(TGprop.getLocalGroupNumber(),TG.Global().rank()));
  auto Qcomm_roots(Qcomm.split(TGprop.Node().rank(),Qcomm.rank()));
  {
    CMatrix lik({NMO,NMO});
    CMatrix lak({nup,NMO});
    for(int nd=0; nd<ndet; nd++) {
      // all nodes accross Qcomm share same segment {nc0,ncN}
      for(int nc=0; nc<local_ncv; nc++) {
        if( nc%Qcomm.size() != Qcomm.rank() ) continue;  
        for(int i=0, ik=0; i<NMO; i++)
          for(int k=0; k<NMO; k++, ik++)
            lik[i][k] = ComplexType(static_cast<RealType>(Likn[ik][nc]),0.0);
        ma::product(PsiT[nspins*nd], lik, lak);     
        for(int a=0; a<nup; a++) 
          copy_n_cast(lak[a].origin(),NMO,to_address(Lank[nspins*nd][a][nc].origin()));
        if(ndet==1) 
          for(int a=0, ak=0; a<nup; a++) 
            for(int k=0; k<NMO; k++, ak++)
              Lakn[ak][nc] = static_cast<SPComplexType>(lak[a][k]);
        if(type==COLLINEAR) {
          ma::product(PsiT[2*nd+1], lik, lak.sliced(0,ndown));  
          for(int a=0; a<ndown; a++)
            copy_n_cast(lak[a].origin(),NMO,to_address(Lank[2*nd+1][a][nc].origin()));
          if(ndet==1)
            for(int a=0, ak=nup*NMO; a<ndown; a++) 
              for(int k=0; k<NMO; k++, ak++)
                Lakn[ak][nc] = static_cast<SPComplexType>(lak[a][k]);
        }
      }
    }
  }
  TG.Global().barrier();
  if(TG.Node().root()) {
    for(auto& v:Lank) 
      Qcomm_roots.all_reduce_in_place_n(to_address(v.origin()),v.num_elements(),std::plus<>());
    if(ndet==1)
      Qcomm_roots.all_reduce_in_place_n(to_address(Lakn.origin()),
                                       Lakn.num_elements(),std::plus<>());
    std::fill_n(to_address(vn0.origin()),vn0.num_elements(),ComplexType(0.0));
  }
  TG.Node().barrier();

  // calculate vn0(i,l) = -0.5 sum_j sum_n L[i][j][n] L[j][l][n] = -0.5 sum_j sum_n L[i][j][n] L[l][j][n]
  {
    int i0,iN;
    std::tie(i0,iN) = FairDivideBoundary(Qcomm.rank(),NMO,Qcomm.size());
    if(iN>i0) {
      SpRMatrix v_({iN-i0,NMO});
      SpRMatrix_ref Lijn(to_address(Likn.origin()),{NMO,NMO*local_ncv});
      ma::product(-0.5,Lijn.sliced(i0,iN),ma::T(Lijn),0.0,v_);
      copy_n_cast( v_.origin(), v_.num_elements(), to_address(vn0[i0].origin()));
    }
  }
  TG.Node().barrier();

  if( TG.Node().root() ) {
    Qcomm_roots.all_reduce_in_place_n(to_address(vn0.origin()),
                                       vn0.num_elements(),std::plus<>());
    dump.pop();
    dump.close();
  }

  if(TG.TG_local().size() > 1 || not (batched=="yes" || batched == "true" ))
    return HamiltonianOperations(Real3IndexFactorization(TGwfn,type,std::move(H1),std::move(haj),
            std::move(Likn),std::move(Lakn),std::move(Lank),std::move(vn0),E0,nc0,global_ncvecs));
  else
    return HamiltonianOperations(Real3IndexFactorization_batched(type,std::move(H1),std::move(haj),
            std::move(Likn),std::move(Lakn),std::move(Lank),std::move(vn0),E0,device_allocator<ComplexType>{},
            nc0,global_ncvecs));

}

#else

HamiltonianOperations RealDenseHamiltonian::getHamiltonianOperations(bool pureSD,
             bool addCoulomb, WALKER_TYPES type, std::vector<PsiT_Matrix>& PsiT,
             double cutvn, double cutv2, TaskGroup_& TGprop, TaskGroup_& TGwfn,
             hdf_archive& hdf_restart) {
  APP_ABORT(" Error: RealDenseHamiltonian requires complex build (QMC_COMPLEX=1). \n"); 
  return HamiltonianOperations();
}


#endif

}
}