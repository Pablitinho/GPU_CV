/*
 * TCVStereo.cpp
 *
 *  Created on: 06/03/2015
 *      Author: pablo
 */

#include "TCVStereo.h"
#include "CVCudaUtils.cuh"
#include "cuda_fp16.h"
#include "device_launch_parameters.h"
//==========================================================================
__global__ void Compute_PEigen_Kernel(half * MemIm1,half * MemG,half * MemU,half * MemP1x,half *MemP2x,half * MemAPP1x,half *MemAPP2x,float Tau,float epsi,bool FirsTime,half *MNx,half *MNy,int Width,int Height)
{
   //===============================================================================================
   //
   //===============================================================================================
   int globalX = (blockIdx.x * blockDim.x + threadIdx.x);
   int globalY = (blockIdx.y * blockDim.y + threadIdx.y);

   int GlobalOffset = (globalY * Width + globalX);
   //===============================================================================================
   if (globalX>1 && globalY>1 && globalX<Width-2 && globalY<Height-2)
   {
	   float P_X=0,P_Y=0;
	   //----------------------------------------------------------------
	   float u_x=0,u_y=0,W=0,S=0,Center=0;
	   //----------------------------------------------------------------
	   // u_x u_y
	   //----------------------------------------------------------------
       if (!FirsTime)
       {
		   W=0.0f;
		   S=0.0f;
		   Center=__half2float(MemU[GlobalOffset]);
		   W=__half2float(MemU[GlobalOffset-1]);
		   S=__half2float(MemU[GlobalOffset+Width]);
		   u_x=W-Center;
		   u_y=S-Center;
       }
       else
       {
    	   u_x=0;
    	   u_y=0;
       }

	   //----------------------------------------------------------------
	   // I_x I_y
	   //----------------------------------------------------------------
/*
	   W=0.0f;
	   S=0.0f;
	   Center=__half2float(MemIm1[GlobalOffset]);
	   //----------------------------------------------------------------
	   Out=0;
	   if (globalX>=1)
	   {
	       W=__half2float(MemIm1[GlobalOffset-1]);
	   }
	   else  W=__half2float(MemIm1[GlobalOffset]);

	   if (globalY<(Height-1))
	   {
	       S=__half2float(MemIm1[GlobalOffset+Width]);
	   }
	   else S=__half2float(MemIm1[GlobalOffset]);


	   //----------------------------------------------------------------
	   float UH,UV;
	   UH=W-Center;
       UV=S-Center;

	   float Magnitud=sqrt(UH*UH + UV*UV);

       //float Nx=UH/(Magnitud+ 0.00000002);
       //float Ny=UV/(Magnitud+ 0.00000002);
*/
	   float Nx=__half2float(MNx[GlobalOffset]);
	   float Ny=-__half2float(MNy[GlobalOffset]);

	   float G_Value=__half2float(MemG[GlobalOffset]);

	   float TmpX=0,TmpY=0;
	   float Reprojection=0;

	   float alfa = 0.85;//Ori 0.85
	   //----------------------------------------------------------------
	   // PP1X PP2X
	   //----------------------------------------------------------------
	   float PP1=u_x*((G_Value*(Nx*Nx)+alfa*Ny*Ny))+u_y*((G_Value*Nx*Ny-alfa*(Nx*Ny)));
       float PP2=u_y*((G_Value*(Ny*Ny)+alfa*Nx*Nx))+u_x*((G_Value*Nx*Ny-alfa*(Nx*Ny)));
	   //----------------------------------------------------------------
       if (!FirsTime)
       {
		   P_X=__half2float(MemP1x[GlobalOffset]);
	       P_Y=__half2float(MemP2x[GlobalOffset]);
	   }
       else
	   {
		   P_X=0.0;
		   P_Y=0.0;
	   }
	   //--------------------------
	   TmpX=P_X+Tau*(PP1-epsi*P_X);
       TmpY=P_Y+Tau*(PP2-epsi*P_Y);
       //--------------------------
       Reprojection = max(1.0, sqrt(TmpX*TmpX + TmpY*TmpY + 0.00002));
	   TmpX=TmpX/Reprojection;
	   TmpY=TmpY/Reprojection;
	   //--------------------------
	   MemP1x[GlobalOffset]=__float2half(TmpX);
	   MemP2x[GlobalOffset]=__float2half(TmpY);
	   ReplicatePixels(MemP1x, TmpX,globalX, globalY,GlobalOffset,Width, Height);
	   ReplicatePixels(MemP2x, TmpY,globalX, globalY,GlobalOffset,Width, Height);
	   //--------------------------
	   float app1=TmpX*((G_Value*(Nx*Nx)+alfa*Ny*Ny))+TmpY*((G_Value*Nx*Ny-alfa*(Nx*Ny)));
	   float app2=TmpY*((G_Value*(Ny*Ny)+alfa*Nx*Nx))+TmpX*((G_Value*Nx*Ny-alfa*(Nx*Ny)));
	   //--------------------------
	   MemAPP1x[GlobalOffset]=__float2half(app1);
	   MemAPP2x[GlobalOffset]=__float2half(app2);
	   //--------------------------
	   ReplicatePixels(MemAPP1x, app1,globalX, globalY,GlobalOffset,Width, Height);
	   ReplicatePixels(MemAPP2x, app2,globalX, globalY,GlobalOffset,Width, Height);
   }
   else
   {
	   if (globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
	   {
		   //-------------------------------------------
		   MemP1x[GlobalOffset]=__float2half(0.0f);
		   MemP2x[GlobalOffset]=__float2half(0.0f);
		   //-------------------------------------------
		   //MemAPP1x[GlobalOffset]=__float2half_rn(0.0f);
		   //MemAPP2x[GlobalOffset]=__float2half_rn(0.0f);
		   //-------------------------------------------
	   }
   }
}
//==========================================================================
__global__ void Update_OF_Up_Kernel(half * MemU,half * MemUp, half * DivU, half * MemIx, half * MemIy, half * MemIt, half *MemU0, float Theta,float Sigma,float Lambda,int FirstTime,int Warped,int Width,int Height)
{
	//===============================================================================================
    //
	//===============================================================================================
	int globalX = (blockIdx.x * blockDim.x + threadIdx.x);
    int globalY = (blockIdx.y * blockDim.y + threadIdx.y);

    int GlobalOffset = (globalY * Width + globalX);
   //===============================================================================================
   if (globalX>1 && globalY>1 && globalX<Width-2 && globalY<Height-2)
   {
	   //---------------------------------------
	   float Ix=0.0;
	   float Iy=0.0;
	   float It=0.0;
	   float u=0.0;
	   float u0=0.0;

	   Ix=__half2float(MemIx[GlobalOffset]);
	   Iy=__half2float(MemIy[GlobalOffset]);
       It=__half2float(MemIt[GlobalOffset]);

	   if (Warped==1)
	   {
         u=__half2float(MemU[GlobalOffset]);

		 MemU0[GlobalOffset]=__float2half(u);

		 u0=u;
	   }
	   else
	   {
	     u=__half2float(MemU[GlobalOffset]);

		 if (!FirstTime)
		 {
			 u0=__half2float(MemU0[GlobalOffset]);
		 }
		 else
		 {
			 u0=0.0f;
		 }
	   }
       //---------------------------------------
	   float nup=0;
	   float up=0;

	   float I_Grad=  (Ix*Ix+Iy*Iy+0.0000001);//OF
	   //float I_Grad=  (Ix*Ix+0.00000001);//STEREO

	   float Rho;
	   float Umbral;


	   Rho=(It + (u-u0)*Ix);//STEREO
	   //Rho=(It + (u-u0)*Ix + (v-v0)*Iy + 0.00000001);//OF
	   //Rho=Rho*2;
	   Umbral=Sigma*Lambda*I_Grad;

	   up=0.0f;
	   //-------------------------------------------------
	   if (Rho<-Umbral)
	   {
		   up=Sigma*Lambda*Ix;
	   }
	   //-------------------------------------------------
	   else if (Rho>Umbral)
	   {
		   up=-Sigma*Lambda*Ix;
	   }
	   //-------------------------------------------------
	   else if (abs(Rho)<=Umbral)
	   {
		   up=-(Rho*Ix)/I_Grad;//OF
	   }
	   //-------------------------------------------------
	   nup=u+up;

	   if (globalX<=1 || globalX>=Width-2 || globalY<=1 || globalY>=Height-2)
	   {
			nup = 0.0;
	   }

	   //------------------------------------------
	   MemUp[GlobalOffset]=__float2half(nup);
	   //----------------------------------------------------------------
	   // Update
	   //----------------------------------------------------------------
	   float DivergenceU=0.0f;

	   if (!FirstTime)
	   {
	       DivergenceU=__half2float(DivU[GlobalOffset]);
	   }

	   float du=Theta*DivergenceU;
	   //----------------------------------------------------------------
	   u=nup+du;
	   //----------------------------------------------------------------
	   if (isnan(u) || abs(u)<0.01|| u>0)
	   {
	       u=0.0;
	   }
	   MemU[GlobalOffset]=__float2half(u);
       //------------------------------------------
   }
   else
   {
	   if(globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
	   {
	 	   MemU[GlobalOffset]=__float2half(0.0f);
		   MemUp[GlobalOffset]=__float2half(0.0f);
	       MemU0[GlobalOffset]=__float2half(0.0f);
	   }
   }
}
//==========================================================================
TCVStereo::TCVStereo(void * d_Gpu)
{
	Gpu = d_Gpu;
	//-----------------------------------------------

	MemPyrIm1=NULL;
	MemPyrIm2=NULL;

	MemUPrev=NULL;
	MemVPrev=NULL;

    MemPyrIm2Warped=NULL;

	MemPyrG=NULL;

	MemP1=NULL;
	MemP2=NULL;

	MemAPP1=NULL;
	MemAPP2=NULL;


	MemDivx=NULL;

	MemCensus1=NULL;
	MemCensus2=NULL;

	MemIx=NULL;
	MemIy=NULL;
    MemIt=NULL;

	MemU=NULL;

	MemU0=NULL;

	MemUp=NULL;

	MemNx=NULL;
	MemNy=NULL;

	MemHFAux1=NULL;
	MemHFAux2=NULL;
	MemHFAux3=NULL;

	Scales=0;
}
//--------------------------------------------------------------------------
void TCVStereo::InitPyramid(int NumScalesMax,int MinSize,float Factor,uint Width, uint Height)
{
	 for (int i=0;i<Scales;i++)
	 {
		  delete MemUPrev[i];
		  delete MemVPrev[i];

	      delete MemPyrIm1[i];
	      delete MemPyrIm2[i];
	      delete MemPyrIm2Warped[i];
	      delete MemPyrG[i];

	      delete MemP1[i];
	      delete MemP2[i];


	      delete MemAPP1[i];
	      delete MemAPP2[i];

	      delete MemDivx[i];


	      delete MemCensus1[i];
	      delete MemCensus2[i];

	      delete MemIx[i];
	      delete MemIy[i];
	      delete MemIt[i];

	      delete MemU[i];

	      delete MemU0[i];

	      delete MemUp[i];

	      delete MemNx[i];
	      delete MemNy[i];

	      delete MemHFAux1[i];
	      delete MemHFAux2[i];
	      delete MemHFAux3[i];
	 }

	 int NumScales=((TGpu *)Gpu)->CV->Math->MaximumScales(Width,Height,NumScalesMax,MinSize,Factor);

	 MemUPrev= new TGpuMem::TGpuMemHalfFloat*[NumScales];
	 MemVPrev= new TGpuMem::TGpuMemHalfFloat*[NumScales];

     MemPyrIm1 = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemPyrIm2 = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemPyrIm2Warped = new TGpuMem::TGpuMemHalfFloat*[NumScales];

     MemPyrG = new TGpuMem::TGpuMemHalfFloat*[NumScales];

     MemP1 = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemP2 = new TGpuMem::TGpuMemHalfFloat*[NumScales];

     MemAPP1 = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemAPP2 = new TGpuMem::TGpuMemHalfFloat*[NumScales];

     MemDivx = new TGpuMem::TGpuMemHalfFloat*[NumScales];

     MemCensus1 = new TGpuMem::TGpuMemUInt*[NumScales];
     MemCensus2 = new TGpuMem::TGpuMemUInt*[NumScales];

     MemIx = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemIy = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemIt = new TGpuMem::TGpuMemHalfFloat*[NumScales];

     MemU = new TGpuMem::TGpuMemHalfFloat*[NumScales];

     MemU0 = new TGpuMem::TGpuMemHalfFloat*[NumScales];

     MemUp = new TGpuMem::TGpuMemHalfFloat*[NumScales];

     MemNx = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemNy = new TGpuMem::TGpuMemHalfFloat*[NumScales];

     MemHFAux1 = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemHFAux2 = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemHFAux3 = new TGpuMem::TGpuMemHalfFloat*[NumScales];

	 for (int i=0;i<NumScales;i++)
	 {
		 MemUPrev[i]= new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemUPrev[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemVPrev[i]= new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemVPrev[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemPyrIm1[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemPyrIm1[i]->Init((unsigned char)0);
		 //---------------------------------------------------------------------------
		 MemPyrIm2[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemPyrIm2[i]->Init((unsigned char)0);
		 //---------------------------------------------------------------------------
		 MemPyrIm2Warped[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemPyrIm2Warped[i]->Init((unsigned char)0);
		 //---------------------------------------------------------------------------
		 MemPyrG[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemPyrG[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemP1[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemP1[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemP2[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemP2[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemAPP1[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemAPP1[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemAPP2[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemAPP2[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemDivx[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemDivx[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemCensus1[i] = new TGpuMem::TGpuMemUInt(Gpu,(uint)Width,(uint)Height,1, false);
		 MemCensus1[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemCensus2[i] = new TGpuMem::TGpuMemUInt(Gpu,(uint)Width,(uint)Height,1, false);
		 MemCensus2[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemIx[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemIx[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemIy[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemIy[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemIt[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemIt[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemU[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemU[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemU0[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemU0[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemUp[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemUp[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemNx[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemNx[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemNy[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemNy[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemHFAux1[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemHFAux1[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemHFAux2[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemHFAux2[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemHFAux3[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemHFAux3[i]->Init(0);
		 //---------------------------------------------------------------------------

		 Width=(uint)(Width*Factor);
		 Height= (uint)(Height*Factor);

	 }

	 Scales= NumScales;
}
//--------------------------------------------------------------------------
void TCVStereo::InitPyramid()
{
	 if (Scales!=0)
	 {
		 for (int i=0;i<Scales;i++)
		 {
			 //---------------------------------------------------------------------------
			 MemUPrev[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemVPrev[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemPyrIm1[i]->Init((unsigned char)0);
			 //---------------------------------------------------------------------------
			 MemPyrIm2[i]->Init((unsigned char)0);
			 //---------------------------------------------------------------------------
			 MemPyrIm2Warped[i]->Init((unsigned char)0);
			 //---------------------------------------------------------------------------
			 MemPyrG[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemP1[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemP2[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemAPP1[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemAPP2[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemDivx[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemCensus1[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemCensus2[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemIx[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemIy[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemIt[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemU[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemU0[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemUp[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemNx[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemNy[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemHFAux1[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemHFAux2[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemHFAux3[i]->Init(0);
			 //---------------------------------------------------------------------------
		 }
	 }
}
//--------------------------------------------------------------------------
void TCVStereo::GeneratePyramid(TGpuMem::TGpuMemUChar  * MemSrc,TGpuMem::TGpuMemHalfFloat  ** MemPyramid)
{
	 if (*MemPyramid!=NULL)
	 {
		 //MemSrc->Copy(MemPyramid[0]);
		 MemSrc->Casting(MemPyramid[0]);
		 for (int i=0;i<Scales-1;i++)
		 {
			 ((TGpu *)Gpu)->CV->Geometry->ResizeBilinear(MemPyramid[i],MemPyramid[i+1]);
		 }
	 }
}
//--------------------------------------------------------------------------
void TCVStereo::GeneratePyramid(TGpuMem::TGpuMemHalfFloat  * MemSrc,TGpuMem::TGpuMemHalfFloat  ** MemPyramid)
{
	 if (*MemPyramid!=NULL)
	 {
		 MemSrc->Copy(MemPyramid[0]);
		 for (int i=0;i<Scales-1;i++)
		 {
			 ((TGpu *)Gpu)->CV->Geometry->ResizeBilinear(MemPyramid[i],MemPyramid[i+1]);
		 }
	 }
}
//--------------------------------------------------------------------------
void TCVStereo::GeneratePyramidOF(TGpuMem::TGpuMemHalfFloat  * MemSrc,TGpuMem::TGpuMemHalfFloat  ** MemPyramid)
{
	 if (*MemPyramid!=NULL)
	 {
		 MemSrc->Copy(MemPyramid[0]);
		 for (int i=0;i<Scales-1;i++)
		 {
			 ((TGpu *)Gpu)->CV->Math->Div(MemPyramid[i],2.0f,MemHFAux2[i]);
			 ((TGpu *)Gpu)->CV->Geometry->ResizeBilinear(MemHFAux2[i],MemPyramid[i+1]);
		 }
	 }
}
//--------------------------------------------------------------------------
// Standard Method
//--------------------------------------------------------------------------
void TCVStereo::AniTVL1_Stereo(TGpuMem::TGpuMemHalfFloat *MemSrc1,TGpuMem::TGpuMemHalfFloat *MemSrc2,TGpuMem::TGpuMemHalfFloat *U,int NumIters,int NumWarps,float Alpha, float Beta,bool PyramidalIter)
{

	 GeneratePyramid(MemSrc1,MemPyrIm1);
	 GeneratePyramid(MemSrc2,MemPyrIm2);
	 //----------------------------------------------
     // Compute Optical Flow per each Scale
     //----------------------------------------------
	 float FactorX;// , FactorY;
     for (int i = Scales - 1; i >= 0; i--)
     {
         //----------------------------------------------
         // Compute OF
         //----------------------------------------------
         Compute_OF_TV_L1_Huber(i,NumIters, NumWarps, Alpha, Beta, PyramidalIter);
         //----------------------------------------------
         // Resize U,V
         //----------------------------------------------
         if (i != 0)
         {
             //-----------------------------------
        	 ((TGpu *)Gpu)->CV->Geometry->ResizeBilinear(MemU[i], MemU[i - 1]);
             //-----------------------------------
             FactorX = (float)MemU[i - 1]->Width() / (float)MemU[i]->Width();
             //FactorY = (float)MemU[i - 1]->Height() / (float)MemU[i]->Height();
             ((TGpu *)Gpu)->CV->Math->Mult(MemU[i - 1], FactorX, MemU[i - 1]);
             //-----------------------------------
         }
     }
     //----------------------------------------------
     MemU[0]->Copy(U);
}
//--------------------------------------------------------------------------
void TCVStereo::Compute_OF_TV_L1_Huber(int NumScale, int NumIters, int NumWarps, float Alpha, float Beta, bool PiramidalIteration)
{
    //---------------------------------------------------------------------------------
    bool FirstTime = true;
    int Warped = 1;
    float Theta;
    float Sigma;
    float a = 3.5f;
    float SST;
    //---------------------------------------------------------------------------------
	// Compute Census & Diffusion Weight
    //---------------------------------------------------------------------------------
	((TGpu *)Gpu)->CV->Features->DiffusionWeight(MemPyrIm1[NumScale],MemPyrG[NumScale],Alpha,Beta);
	((TGpu *)Gpu)->CV->Features->Census(MemPyrIm1[NumScale], MemCensus1[NumScale],4);
    //---------------------------------------------------------------------------------
	if (MemPyrIm1[NumScale]->Width()<160 || MemPyrIm1[NumScale]->Height()<160)
	    ((TGpu *)Gpu)->CV->Features->EigenVectors(MemPyrIm1[NumScale],MemNx[NumScale],MemNy[NumScale],((TGpu *)Gpu)->CV->Filters->d_FilterGauss_1);
	else
		((TGpu *)Gpu)->CV->Features->EigenVectors(MemPyrIm1[NumScale],MemNx[NumScale],MemNy[NumScale],((TGpu *)Gpu)->CV->Filters->d_FilterGauss_2);

    if (true)
    {
        float Factor = ((float)(NumScale-1 ) / (float)(Scales-1));
        NumIters = (int)(((1.0 - Factor) * NumIters) + (Factor * ((float)NumIters / 1.9f)));
        if (NumIters <= 1)
        {
            NumIters = 5;
        }
    }
    if (NumScale==Scales-1)
    {
    	MemU[NumScale]->Init(0.0f);
    }
    MemDivx[NumScale]->Init(0.0f);

    NumWarps= (int)ceil(NumIters/NumWarps)+1;
    //---------------------------------------------------------------------------------
	for (int iter = 0; iter < NumIters; iter++)
    {
        SST = (float)sin(((((float)(iter+4) / a) - (floor(0.5 + ((float)(iter+4) / a)))) + 0.5) + 0.6);
        //Theta = SST * 0.7f;
        Sigma = SST * 0.9f;

        Theta = 0.4f;
        Sigma = 0.5f;
        //-------------------------------------------------------------------
        if (iter % NumWarps == 0 && iter != (NumIters - 1))
        {
        	Compute_Warping(NumScale);
        	Warped=1;
        }
        //-------------------------------------------------------------------
        //for (int i=0;i<5;i++)
        Compute_PEigen(NumScale,FirstTime);

        ((TGpu *)Gpu)->CV->Math->Divergence(MemAPP1[NumScale], MemAPP2[NumScale],MemDivx[NumScale]);
        //-------------------------------------------------------------------
        Update_OF_Up(NumScale,FirstTime,Theta,Sigma,Warped);
        ((TGpu *)Gpu)->CV->Utils->ReplicateEdges(MemU[NumScale],MemU[NumScale]);
        //-------------------------------------------------------------------
/*
        if (FirstTime)
        {
           Update_OF_Up_Vp(NumScale,FirstTime,Theta,Sigma,Warped);
           Compute_PEigen(NumScale,FirstTime);
           //-------------------------------------------------------------------
           ((TGpu *)Gpu)->CV->Math->Divergence(MemAPP1[NumScale], MemAPP2[NumScale],MemDivx[NumScale]);
           ((TGpu *)Gpu)->CV->Math->Divergence(MemAPP3[NumScale], MemAPP4[NumScale],MemDivy[NumScale]);
           //-------------------------------------------------------------------
           Update_OF_Up_Vp(NumScale,false,Theta,Sigma,Warped);
           ((TGpu *)Gpu)->CV->Utils->ReplicateEdges(MemU[NumScale],MemV[NumScale]);
        }
        //-------------------------------------------------------------------
        //Iter_Compute_PEigen(NumScale,FirstTime);

        for (int i=0;i<2;i++)
        {
        	Theta = 0.5f;
            Compute_PEigen(NumScale,FirstTime);
        }
        //-------------------------------------------------------------------
        ((TGpu *)Gpu)->CV->Math->Divergence(MemAPP1[NumScale], MemAPP2[NumScale],MemDivx[NumScale]);
        ((TGpu *)Gpu)->CV->Math->Divergence(MemAPP3[NumScale], MemAPP4[NumScale],MemDivy[NumScale]);
        //-------------------------------------------------------------------
        Update_OF_Up_Vp(NumScale,FirstTime,Theta,Sigma,Warped);

        ((TGpu *)Gpu)->CV->Utils->ReplicateEdges(MemU[NumScale],MemV[NumScale]);
*/
        FirstTime=false;
        Warped=0;

    }
    //---------------------------------------------------------------------------------
	// Median Filter
    //---------------------------------------------------------------------------------
	((TGpu *)Gpu)->CV->Filters->Median3x3(MemU[NumScale],MemHFAux1[NumScale]);
	MemHFAux1[NumScale]->Copy(MemU[NumScale]);
    //---------------------------------------------------------------------------------
}
//--------------------------------------------------------------------------
//Core
//--------------------------------------------------------------------------
void TCVStereo::Update_OF_Up(int NumScale,int FirstTime,float Theta,float Sigma,int Warped)
{
    float Lambda = 125.0f;
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp((int)(MemPyrIm1[NumScale]->Width()), numThreads.x), ((TGpu *)Gpu)->iDivUp((int)(MemPyrIm1[NumScale]->Height()), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Update_OF_Up_Kernel<<<numBlocks, numThreads>>>(MemU[NumScale]->GetMemory(),MemUp[NumScale]->GetMemory(),MemDivx[NumScale]->GetMemory(),MemIx[NumScale]->GetMemory(),MemIy[NumScale]->GetMemory(),MemIt[NumScale]->GetMemory(),MemU0[NumScale]->GetMemory(),Theta, Sigma, Lambda, FirstTime, Warped,MemPyrIm1[NumScale]->Width(), MemPyrIm1[NumScale]->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVStereo::Compute_PEigen(int NumScale,bool FirstTime)
{
    float Tau = 0.5f;
    float Epsilon = 0.0001f;
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemPyrIm1[NumScale]->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemPyrIm1[NumScale]->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Compute_PEigen_Kernel<<<numBlocks, numThreads>>>(MemPyrIm1[NumScale]->GetMemory(),MemPyrG[NumScale]->GetMemory(), MemU[NumScale]->GetMemory(), MemP1[NumScale]->GetMemory(),MemP2[NumScale]->GetMemory(),MemAPP1[NumScale]->GetMemory(),MemAPP2[NumScale]->GetMemory(),Tau,Epsilon,FirstTime,MemNx[NumScale]->GetMemory(),MemNy[NumScale]->GetMemory(),MemPyrIm1[NumScale]->Width(), MemPyrIm1[NumScale]->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVStereo::Compute_Warping(int NumScale)
{
    //---------------------------------------------------------------------------------
	// Median Filter
    //---------------------------------------------------------------------------------
	((TGpu *)Gpu)->CV->Filters->Median3x3(MemU[NumScale],MemHFAux1[NumScale]);
	MemHFAux1[NumScale]->Copy(MemU[NumScale]);
    //---------------------------------------------------------------------------------
	MemHFAux1[NumScale]->Init(0);
	((TGpu *)Gpu)->CV->Geometry->Warping(MemPyrIm2[NumScale],MemPyrIm2Warped[NumScale],MemU[NumScale],MemHFAux1[NumScale],true);
	((TGpu *)Gpu)->CV->Features->Census(MemPyrIm2Warped[NumScale], MemCensus2[NumScale],4);
	((TGpu *)Gpu)->CV->Filters->CensusDerivates(MemCensus1[NumScale],MemCensus2[NumScale],MemIx[NumScale],MemIy[NumScale],MemIt[NumScale]);
    //---------------------------------------------------------------------------------
}
//--------------------------------------------------------------------------
TCVStereo::~TCVStereo()
{

    for (int i=0;i<Scales;i++)
    {
    	  delete MemUPrev[i];
    	  delete MemVPrev[i];
	      delete MemPyrIm1[i];
	      delete MemPyrIm2[i];
	      delete MemPyrIm2Warped[i];
	      delete MemPyrG[i];

	      delete MemP1[i];
	      delete MemP2[i];

	      delete MemAPP1[i];
	      delete MemAPP2[i];

	      delete MemDivx[i];

	      delete MemCensus1[i];
	      delete MemCensus2[i];

	      delete MemIx[i];
	      delete MemIy[i];
	      delete MemIt[i];

	      delete MemU[i];

	      delete MemU0[i];

	      delete MemUp[i];

	      delete MemNx[i];
	      delete MemNy[i];

	      delete MemHFAux1[i];
	      delete MemHFAux2[i];
	      delete MemHFAux3[i];
    }
}
//--------------------------------------------------------------------------
