/////////////////////////////////////////////////////////////////nspli
//jjjkkkkkk/
// (c) Copyright 2012-  by Jeongnim Kim and Ken Esler           //
//////////////////////////////////////////////////////////////////
/** @file EsinplineAdoptor.h
 *
 * Adoptor classes and BsplineSet<SplineAdoptor> 
 * BsplineSet<SplineAdoptor> is a SPOSetBase class to work with determinant classes
 * SplineAdoptor provides these functions 
 * - evaluate_v    value only
 * - evaluate_vgl  vgl
 * - evaluate_vgh  vgh
 * Specializations are implemented  in Spline*Adoptor.h and include
 * - SplineC2RAdoptor<ST,TT,D> : real wavefunction using complex einspline, tiling
 * - SplineC2CAdoptor<ST,TT,D> : complex wavefunction using complex einspline, tiling
 * - SplineR2RAdoptor<ST,TT,D> : real wavefunction using real einspline, a single twist
 * where ST (TT) is the precision of the einspline (SPOSetBase).
 * 
 * typedefs and data members are duplicated for each adoptor class.
 * @todo Specalization and optimization for orthorhombic cells to use vgl not vgh
 */
#ifndef QMCPLUSPLUS_EINSPLINE_ADOPTOR_H
#define QMCPLUSPLUS_EINSPLINE_ADOPTOR_H

#include <Lattice/CrystalLattice.h>
#include <spline/einspline_engine.hpp>
#include <spline/einspline_util.hpp>

namespace qmcplusplus {

  /** einspline trait class equivalent to  MultiOrbitalTraits
   * @tparam ST spline datatype
   * @tparam D dimension
   * @tparam TT target datatype
   */
  template<typename ST, unsigned D>
    struct einspline_traits { };

  /** specialization for 3D double */
  template<>
    struct einspline_traits<double,3>
    {
      typedef multi_UBspline_3d_d SplineType;  
      typedef UBspline_3d_d       SingleSplineType;  
      typedef BCtype_d            BCType;
      typedef double              DataType;
    };

  /** specialization for 3D complex<double> */
  template<>
    struct einspline_traits<complex<double>,3>
    {
      typedef multi_UBspline_3d_z SplineType;  
      typedef UBspline_3d_z       SingleSplineType;  
      typedef BCtype_z            BCType;
      typedef complex<double>     DataType;
    };

  /** specialization for 3D float */
  template<>
    struct einspline_traits<float,3>
    {
      typedef multi_UBspline_3d_s SplineType;  
      typedef UBspline_3d_s       SingleSplineType;  
      typedef BCtype_s            BCType;
      typedef float               DataType;
    };

  /** specialization for 3D complex<float> */
  template<>
    struct einspline_traits<complex<float>,3>
    {
      typedef multi_UBspline_3d_c SplineType;  
      typedef UBspline_3d_c       SingleSplineType;  
      typedef BCtype_c            BCType;
      typedef complex<float>      DataType;
    };

  //inline void computePhases(const PointType& r)
  //{
  //  for (int i=0; i<kPoints.size(); i++) phase[i] = -dot(r, kPoints[i]);
  //  eval_e2iphi(kPoints.size(),phase.data(),eikr.data());
  //}
 
  /** base class any SplineAdoptor
   *
   * This handles SC and twist
   */
  template<typename ST, unsigned D>
    struct SplineAdoptorBase
    {
      TinyVector<int,D>          HalfG;
      vector<bool>               MakeTwoCopies;
      Tensor<ST,D>               GGt;
      vector<TinyVector<ST,D> >  kPoints;
      CrystalLattice<ST,D>       SuperLattice;
      CrystalLattice<ST,D>       PrimLattice;

      ///first index of the SPOs this Spline handles
      int FirstIndex;
      ///last index of the SPOs this Spline handles
      int LastIndex;

      SplineAdoptorBase():FirstIndex(0),LastIndex(0)
      {
      }

      inline void init_base(int n)
      {
        GGt=dot(transpose(PrimLattice.G),PrimLattice.G);
        kPoints.resize(n);
        MakeTwoCopies.resize(n);
      }
    };

  /** a class to map a memory sequence to a vector
   * @tparam T datatype
   */
  template<typename T>
    struct VectorViewer
    {
      T* data_ptr;
      int data_size;
      inline VectorViewer(T* a, int n):data_ptr(a),data_size(n){}
      inline T* data() { return data_ptr;}
      inline int size() const { return data_size;}
      inline T& operator[](int i) { return data_ptr[i]; }
      inline T operator[](int i) const { return data_ptr[i]; }
    };


  /** BsplineSet<SplineAdoptor>, a SPOSetBase
   * @tparam SplineAdoptor implements evaluation functions that matched the storage requirements.
   *
   * Equivalent to EinsplineSetExtended<Storage>
   * Storage is now handled by SplineAdoptor class that is specialized for precision, storage etc.
   */
  template<typename SplineAdoptor>
    struct BsplineSet: public SPOSetBase, public SplineAdoptor
  {
    typedef typename SplineAdoptor::SplineType SplineType;
    typedef typename SplineAdoptor::PointType  PointType;

    using SplineAdoptor::MultiSpline;

    /** default constructor */
    BsplineSet() { }

    void allocate(TinyVector<int,DIM>& mesh, int nv)
    {
      SplineAdoptor::create_spline(mesh,nv);
    }

    /** allocate einspline 
     *
     * A general allocation scheme for a spline
     */
    template<typename GT, typename BCT>
      void allocate(GT& xyz_g, BCT& xyz_bc, int nv)
      {
        SplineType* dummy=0;
        MultiSpline=einspline::create(dummy,xyz_g,xyz_bc,nv);
      }

    SPOSetBase* makeClone() const
    {
      return new BsplineSet<SplineAdoptor>(*this);
    }

    inline void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi)
    {
      SplineAdoptor::evaluate_v(P.R[iat],psi);
    }

    inline void evaluate(const ParticleSet& P, int iat, 
        ValueVector_t& psi, GradVector_t& dpsi, ValueVector_t& d2psi)
    {
      SplineAdoptor::evaluate_vgl(P.R[iat],psi,dpsi,d2psi);
    }
    inline void evaluate(const ParticleSet& P, int iat,
        ValueVector_t& psi, GradVector_t& dpsi, HessVector_t& grad_grad_psi)
    {
      SplineAdoptor::evaluate_vgh(P.R[iat],psi,dpsi,grad_grad_psi);
    }

    void resetParameters(const opt_variables_type& active)
    { }

    void resetTargetParticleSet(ParticleSet& e)
    { }

    void setOrbitalSetSize(int norbs)
    {
      OrbitalSetSize = norbs;
      BasisSetSize=norbs;
    }

    void evaluate_notranspose(const ParticleSet& P, int first, int last
        , ValueMatrix_t& logdet, GradMatrix_t& dlogdet, ValueMatrix_t& d2logdet)
    {
      typedef ValueMatrix_t::value_type value_type;
      typedef GradMatrix_t::value_type grad_type;
      //const int N=last-first;

      for(int iat=first, i=0; iat<last; ++iat,++i)
      {
        VectorViewer<value_type> v(logdet[i],OrbitalSetSize);
        VectorViewer<grad_type> g(dlogdet[i],OrbitalSetSize);
        VectorViewer<value_type> l(d2logdet[i],OrbitalSetSize);
        SplineAdoptor::evaluate_vgl(P.R[iat],v,g,l);
      }
    }
  };

  /** base class to read data and manage spline tables
   *
   * Each SplineAdoptor needs a reader
   */
  struct BsplineReaderBase
  {
    EinsplineSetBuilder* mybuilder;
    Communicate* myComm;

    BsplineReaderBase(EinsplineSetBuilder* e):mybuilder(e)
    { 
      myComm=mybuilder->getCommunicator();
    }

    /** copy minimal informatino from EinsplineSet to manage SplineAdoptor
     */
    template<typename SPE>
      void init(EinsplineSet* in, SPE* out)
      {
        out->PrimLattice=in->PrimLattice;
        out->SuperLattice=in->SuperLattice;
        out->GGt=in->GGt;
        out->setOrbitalSetSize(in->getOrbitalSetSize());
      }

    /** return the path name in hdf5
     */
    inline string psi_g_path(int ti, int spin, int ib)
    {
      ostringstream path;
      path << "/electrons/kpoint_" << ti   
        << "/spin_" << spin << "/state_" << ib << "/psi_g";
      return path.str();
    }

    /** return the path name in hdf5
     */
    inline string psi_r_path(int ti, int spin, int ib)
    {
      ostringstream path;
      path << "/electrons/kpoint_" << ti   
        << "/spin_" << spin << "/state_" << ib << "/psi_r";
      return path.str();
    }

    /** read/bcast psi_g 
     * @param ti twist index
     * @param spin spin index
     * @param ib band index
     * @param cG psi_g as stored in hdf5
     */
    void get_psi_g(int ti, int spin, int ib, Vector<complex<double> >& cG)
    {
      int ncg=0;
      if(myComm->rank()==0)
      {
        string path=psi_g_path(ti,spin,ib);
        HDFAttribIO<Vector<complex<double> > >  h_cG(cG);
        h_cG.read (mybuilder->H5FileID, path.c_str());
        ncg=cG.size();
      }
      myComm->bcast(ncg);
      if(ncg != mybuilder->Gvecs[ti].size())
      {
        APP_ABORT("Failed : ncg != Gvecs[ti].size()");
      }
      myComm->bcast(cG);
    }

    virtual ~BsplineReaderBase(){}

    /** create the actual spline sets
     */
    virtual SPOSetBase* create_spline_set(int spin, EinsplineSet* orbitalset)=0;
  };

}
#endif