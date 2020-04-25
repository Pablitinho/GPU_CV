/*
 * TCVMotion.cpp
 *
 *  Created on: 28/02/2015
 *      Author: pablo
 */

#include "TCVMotion.h"
#include <math.h>
#include <stdio.h>
#include "CVCudaUtils.cuh"
#include "cuda_fp16.h"
#include "device_launch_parameters.h"
using namespace std;

//==========================================================================

// Kernels

//==========================================================================
__global__ void Compute_PEigen_Kernel(half * MemIm1,half * MemG,half * MemU,half * MemV,half * MemP1x,half *MemP2x,half *MemP1y,half *MemP2y,half * MemAPP1x,half *MemAPP2x,half *MemAPP1y,half *MemAPP2y,float Tau,float epsi,bool FirsTime,half *MNx,half *MNy,int Width,int Height)
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
	   float u_x=0,u_y=0,v_x=0,v_y=0,W=0,S=0,Center=0;
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
	   // v_x v_y
	   //----------------------------------------------------------------
       if (!FirsTime)
       {
    	   //float Out;
		   W=0.0f;
		   S=0.0f;
		   //Out=0.0f;
		   Center=__half2float(MemV[GlobalOffset]);
		   W=__half2float(MemV[GlobalOffset-1]);
		   S=__half2float(MemV[GlobalOffset+Width]);
		   v_x=W-Center;
		   v_y=S-Center;
       }
       else
       {
    	   v_x=0;
    	   v_y=0;
       }

	   //----------------------------------------------------------------
	   // I_x I_y
	   //----------------------------------------------------------------
/*
	   W=0.0f;

	   S=0.0f;
	   Center=__half2float(MemIm1[GlobalOffset]);
	   //----------------------------------------------------------------

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

       float Nx=UH/(Magnitud+ 0.00000002);
       float Ny=UV/(Magnitud+ 0.00000002);
*/

       //Ori
       //float Nx=__half2float(MNx[GlobalOffset]);
	   //float Ny=-__half2float(MNy[GlobalOffset]);

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
	   //--------------------------
	   //ReplicatePixels(MemP1x, TmpX,GlobalOffset,Width, Height);
	   //ReplicatePixels(MemP2x, TmpY,GlobalOffset,Width, Height);
	   //--------------------------
	   float app1=TmpX*((G_Value*(Nx*Nx)+alfa*Ny*Ny))+TmpY*((G_Value*Nx*Ny-alfa*(Nx*Ny)));
	   float app2=TmpY*((G_Value*(Ny*Ny)+alfa*Nx*Nx))+TmpX*((G_Value*Nx*Ny-alfa*(Nx*Ny)));
	   //--------------------------
	   MemAPP1x[GlobalOffset]=__float2half(app1);
	   MemAPP2x[GlobalOffset]=__float2half(app2);
	   //--------------------------
	   //ReplicatePixels(MemAPP1x, app1,globalX, globalY,GlobalOffset,Width, Height);
	   //ReplicatePixels(MemAPP2x, app2,globalX, globalY,GlobalOffset,Width, Height);
	   //----------------------------------------------------------------
	   // PP1Y PP2Y
	   //----------------------------------------------------------------
	   PP1=v_x*((G_Value*(Nx*Nx)+alfa*Ny*Ny))+v_y*((G_Value*Nx*Ny-alfa*(Nx*Ny)));
       PP2=v_y*((G_Value*(Ny*Ny)+alfa*Nx*Nx))+v_x*((G_Value*Nx*Ny-alfa*(Nx*Ny)));
	   //----------------------------------------------------------------
       if (!FirsTime)
       {
		   P_X=__half2float(MemP1y[GlobalOffset]);
	       P_Y=__half2float(MemP2y[GlobalOffset]);
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
	   MemP1y[GlobalOffset]=__float2half(TmpX);
	   MemP2y[GlobalOffset]=__float2half(TmpY);
	   //--------------------------
	   //ReplicatePixels(MemP1y, TmpX,GlobalOffset,Width, Height);
	   //ReplicatePixels(MemP2y, TmpY,GlobalOffset,Width, Height);
	   //--------------------------
	   /*MemP1y[GlobalOffset+1]=__float2half_rn(TmpX);
	   MemP2y[GlobalOffset+1]=__float2half_rn(TmpY);
	   MemP1y[GlobalOffset+Width]=__float2half_rn(TmpX);
	   MemP2y[GlobalOffset+Width]=__float2half_rn(TmpY);
	   MemP1y[GlobalOffset+Width+1]=__float2half_rn(TmpX);
	   MemP2y[GlobalOffset+Width+1]=__float2half_rn(TmpY);*/
	   //--------------------------
	   //G_Value=1;
	   //Nx=1;
	   //Ny=1;
	   //TmpX=1;
	   //TmpY=1;
	   app1=TmpX*((G_Value*(Nx*Nx)+alfa*Ny*Ny))+TmpY*((G_Value*Nx*Ny-alfa*(Nx*Ny)));
	   app2=TmpY*((G_Value*(Ny*Ny)+alfa*Nx*Nx))+TmpX*((G_Value*Nx*Ny-alfa*(Nx*Ny)));
	   //--------------------------
	   MemAPP1y[GlobalOffset]=__float2half(app1);
	   MemAPP2y[GlobalOffset]=__float2half(app2);
	   //--------------------------
	   //ReplicatePixels(MemAPP1y, app1,globalX, globalY,GlobalOffset,Width, Height);
	   //ReplicatePixels(MemAPP2y, app2,globalX, globalY,GlobalOffset,Width, Height);
	   //--------------------------
	   //MemAPP1y[GlobalOffset]=__float2half_rn(TmpX*((G_Value*(Nx*Nx)+alfa*Ny*Ny))+TmpY*((G_Value*Nx*Ny-alfa*(Nx*Ny))));
	   //MemAPP2y[GlobalOffset]=__float2half_rn(TmpY*((G_Value*(Ny*Ny)+alfa*Nx*Nx))+TmpX*((G_Value*Nx*Ny-alfa*(Nx*Ny))));
	   //----------------------------------------------------------------
   }
   else
   {
	   if (globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
	   {
		   //-------------------------------------------
		   MemP1x[GlobalOffset]=__float2half(0.0f);
		   MemP2x[GlobalOffset]=__float2half(0.0f);
		   //-------------------------------------------
		   MemAPP1x[GlobalOffset]=__float2half(0.0f);
		   MemAPP2x[GlobalOffset]=__float2half(0.0f);
		   //-------------------------------------------
		   MemP1y[GlobalOffset]=__float2half(0.0f);
		   MemP2y[GlobalOffset]=__float2half(0.0f);
		   //-------------------------------------------
		   MemAPP1y[GlobalOffset]=__float2half(0.0f);
		   MemAPP2y[GlobalOffset]=__float2half(0.0f);
		   //-------------------------------------------
	   }
   }
}
//==========================================================================
__global__ void Update_OF_Up_Vp_Kernel(half * MemU, half * MemV,half * MemUp, half * MemVp, half * DivU, half * DivV, half * MemIx, half * MemIy, half * MemIt, half *MemU0, half *MemV0, float Theta,float Sigma,float Lambda,int FirstTime,int Warped,int Width,int Height)
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
	   float v=0.0;

	   float u0=0.0;
	   float v0=0.0;

	   Ix=__half2float(MemIx[GlobalOffset]);
	   Iy=__half2float(MemIy[GlobalOffset]);
       It=__half2float(MemIt[GlobalOffset]);

	   if (Warped==1)
	   {
         u=__half2float(MemU[GlobalOffset]);
	     v=__half2float(MemV[GlobalOffset]);

		 MemU0[GlobalOffset]=__float2half(u);
	     MemV0[GlobalOffset]=__float2half(v);

		 u0=u;
		 v0=v;
	   }
	   else
	   {
	     u=__half2float(MemU[GlobalOffset]);
	     v=__half2float(MemV[GlobalOffset]);

		 if (!FirstTime)
		 {
			 u0=__half2float(MemU0[GlobalOffset]);
			 v0=__half2float(MemV0[GlobalOffset]);
		 }
		 else
		 {
			 u0=0.0f;
			 v0=0.0f;
		 }

		 //u0=__half2float(MemU0[GlobalOffset]);
		 //v0=__half2float(MemV0[GlobalOffset]);
	   }
       //---------------------------------------
	   float nup=0,nvp=0;
	   float up=0,vp=0;

	   float I_Grad=  (Ix*Ix+Iy*Iy+0.0000001);//OF
	   //float I_Grad=  (Ix*Ix+Gamma*Gamma);//STEREO

	   float Rho;
	   float Umbral;
/*
	   if (Warped==1)
	   {
		   //Rho=(It + 0.5*sign(u)*Ix + Gamma*Gamma);//STEREO
		   Rho=(It + 0.5*sign(u)*Ix + 0.5*sign(v)*Iy + 0.0000001);//OF

		   Umbral=Sigma*Lambda*I_Grad;

		   up=0.0f;
		   vp=0.0f;


		   //-------------------------------------------------
		   if (Rho<-Umbral)
		   {
			   up=Sigma*Lambda*Ix;
			   vp=Sigma*Lambda*Iy;
		   }
		   //-------------------------------------------------
		   else if (Rho>Umbral)
		   {
			   up=-Sigma*Lambda*Ix;
			   vp=-Sigma*Lambda*Iy;
		   }
		   //-------------------------------------------------
		   else if (abs(Rho)<=Umbral)
		   {
			   up=-(Rho*Ix)/I_Grad;//OF
			   //up=-Rho/Ix;//STEREO
			   vp=-Rho*Iy/I_Grad;
		   }
		   u=u+0.5*up;
		   v=v+0.5*vp;
	   }
*/
	   //Rho=(It + (u-u0)*Ix + Gamma*Gamma);//STEREO
	   Rho=(It + (u-u0)*Ix + (v-v0)*Iy + 0.00000001);//OF
	   //Rho=Rho*2;
	   Umbral=Sigma*Lambda*I_Grad;

	   up=0.0f;
	   vp=0.0f;
	   //-------------------------------------------------
	   if (Rho<-Umbral)
	   {
		   up=Sigma*Lambda*Ix;
		   vp=Sigma*Lambda*Iy;
	   }
	   //-------------------------------------------------
	   else if (Rho>Umbral)
	   {
		   up=-Sigma*Lambda*Ix;
		   vp=-Sigma*Lambda*Iy;
	   }
	   //-------------------------------------------------
	   else if (abs(Rho)<=Umbral)
	   {
		   up=-(Rho*Ix)/I_Grad;//OF
		   vp=-(Rho*Iy)/I_Grad;
	   }
	   //-------------------------------------------------
	   nup=u+up;
	   nvp=v+vp;

	   if (globalX<=1 || globalX>=Width-2 || globalY<=1 || globalY>=Height-2)
	   {
			nup = 0.0;
			nvp = 0.0;
	   }
       //nup=2*u-nup;
       //nvp=2*v-nvp;

	   //------------------------------------------
	   MemUp[GlobalOffset]=__float2half(nup);
       MemVp[GlobalOffset]=__float2half(nvp);
       //------------------------------------------
	   /*MemUp[GlobalOffset+1]=__float2half_rn(nup);
       MemVp[GlobalOffset+1]=__float2half_rn(nvp);
	   MemUp[GlobalOffset+Width]=__float2half_rn(nup);
       MemVp[GlobalOffset+Width]=__float2half_rn(nvp);
	   MemUp[GlobalOffset+Width+1]=__float2half_rn(nup);
       MemVp[GlobalOffset+Width+1]=__float2half_rn(nvp);*/
       //------------------------------------------

	   //----------------------------------------------------------------
	   // Update
	   //----------------------------------------------------------------
	   float DivergenceU=0.0f,DivergenceV=0.0f;

	   if (!FirstTime)
	   {
	       DivergenceU=__half2float(DivU[GlobalOffset]);
	       DivergenceV=__half2float(DivV[GlobalOffset]);
	   }

	   float du=Theta*DivergenceU;
	   float dv=Theta*DivergenceV;
	   //----------------------------------------------------------------
	   u=nup+du;
	   v=nvp+dv;
	   //----------------------------------------------------------------
	   /*if ( isnan(u) || isnan(v)|| globalX<1 || globalX>=Width-1 || globalY<1 || globalY>=Height-1)
	   {
	       u=0.0;
		   v=0.0;
	   }*/
	   float Vect_lenght= sqrt(u*u+v*v);
	   if (isnan(u) || isnan(v)/*|| Vect_lenght<0.01 || Vect_lenght>100.01*/)
	   {
	       u=0.0;
		   v=0.0;
	   }
	   MemU[GlobalOffset]=__float2half(u);
	   MemV[GlobalOffset]=__float2half(v);
       //------------------------------------------
	   /*if ((globalX+1)<Width)
	   {
		   MemU[GlobalOffset+1]=__float2half_rn(u);
		   MemV[GlobalOffset+1]=__float2half_rn(v);
	   }
	   if ((globalY+1)<Height)
	   {
		   MemU[GlobalOffset+Width]=__float2half_rn(u);
		   MemV[GlobalOffset+Width]=__float2half_rn(v);
	   }
	   if ((globalX+1)<Width && (globalY+1)<Height)
	   {
		   MemU[GlobalOffset+Width+1]=__float2half_rn(u);
		   MemV[GlobalOffset+Width+1]=__float2half_rn(v);
	   }*/
       //------------------------------------------
	   //ReplicatePixels(MemU, u,globalX,globalY,GlobalOffset,Width, Height);
	   //ReplicatePixels(MemV, v,globalX,globalY,GlobalOffset,Width, Height);
	   //----------------------------------------------------------------
   }
   else
   {
	   if(globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
	   {
	 	   MemU[GlobalOffset]=__float2half(0.0f);
		   MemV[GlobalOffset]=__float2half(0.0f);

		   MemUp[GlobalOffset]=__float2half(0.0f);
	       MemVp[GlobalOffset]=__float2half(0.0f);

	       MemU0[GlobalOffset]=__float2half(0.0f);
	       MemV0[GlobalOffset]=__float2half(0.0f);
	   }
   }
}
//==========================================================================
__global__ void Update_OF_Up_Vp_Prev_Kernel(half * MemU, half * MemV,half *MemG,half * MemUPrev, half * MemVPrev,half * MemUp, half * MemVp, half * DivU, half * DivV, half * MemIx, half * MemIy, half * MemIt, half *MemU0, half *MemV0, float Theta,float Sigma,float Lambda,int FirstTime,float PrevFactor,int Warped,int Width,int Height)
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
	   float v=0.0;

	   float u0=0.0;
	   float v0=0.0;

	   Ix=__half2float(MemIx[GlobalOffset]);
	   Iy=__half2float(MemIy[GlobalOffset]);
       It=__half2float(MemIt[GlobalOffset]);

	   if (Warped==1)
	   {
         u=__half2float(MemU[GlobalOffset]);
	     v=__half2float(MemV[GlobalOffset]);

		 MemU0[GlobalOffset]=__float2half(u);
	     MemV0[GlobalOffset]=__float2half(v);

		 u0=u;
		 v0=v;
	   }
	   else
	   {
	     u=__half2float(MemU[GlobalOffset]);
	     v=__half2float(MemV[GlobalOffset]);

		 if (!FirstTime)
		 {
			 u0=__half2float(MemU0[GlobalOffset]);
			 v0=__half2float(MemV0[GlobalOffset]);
		 }
		 else
		 {
			 u0=0.0f;
			 v0=0.0f;
		 }

	   }
       //---------------------------------------
	   float nup=0,nvp=0;
	   float up=0,vp=0;

	   float I_Grad=  (Ix*Ix+Iy*Iy+0.0000001);//OF

	   float Rho;
	   float Umbral;

	   if (Warped==1)
	   {
		   Rho=(It + 0.5*sign(u)*Ix + 0.5*sign(v)*Iy + 0.0000001);//OF

		   Umbral=Sigma*Lambda*I_Grad;

		   up=0.0f;
		   vp=0.0f;


		   //-------------------------------------------------
		   if (Rho<-Umbral)
		   {
			   up=Sigma*Lambda*Ix;
			   vp=Sigma*Lambda*Iy;
		   }
		   //-------------------------------------------------
		   else if (Rho>Umbral)
		   {
			   up=-Sigma*Lambda*Ix;
			   vp=-Sigma*Lambda*Iy;
		   }
		   //-------------------------------------------------
		   else if (abs(Rho)<=Umbral)
		   {
			   up=-(Rho*Ix)/I_Grad;//OF
			   vp=-Rho*Iy/I_Grad;
		   }
		   u=u+0.5*up;
		   v=v+0.5*vp;
	   }

	   Rho=(It + (u-u0)*Ix + (v-v0)*Iy + 0.00000001);//OF
	   Umbral=Sigma*Lambda*I_Grad;

	   up=0.0f;
	   vp=0.0f;
	   //-------------------------------------------------
	   if (Rho<-Umbral)
	   {
		   up=Sigma*Lambda*Ix;
		   vp=Sigma*Lambda*Iy;
	   }
	   //-------------------------------------------------
	   else if (Rho>Umbral)
	   {
		   up=-Sigma*Lambda*Ix;
		   vp=-Sigma*Lambda*Iy;
	   }
	   //-------------------------------------------------
	   else if (abs(Rho)<=Umbral)
	   {
		   up=-(Rho*Ix)/I_Grad;//OF
		   vp=-(Rho*Iy)/I_Grad;
	   }
	   //-------------------------------------------------
	   nup=u+up;
	   nvp=v+vp;

	   if (globalX<=1 || globalX>=Width-2 || globalY<=1 || globalY>=Height-2)
	   {
			nup = 0.0;
			nvp = 0.0;
	   }
	   //------------------------------------------
	   MemUp[GlobalOffset]=__float2half(nup);
       MemVp[GlobalOffset]=__float2half(nvp);
	   //----------------------------------------------------------------
	   // Update
	   //----------------------------------------------------------------
	   float DivergenceU=0.0f,DivergenceV=0.0f;

	   if (!FirstTime)
	   {
	       DivergenceU=__half2float(DivU[GlobalOffset]);
	       DivergenceV=__half2float(DivV[GlobalOffset]);
	   }

	   float du=Theta*DivergenceU;
	   float dv=Theta*DivergenceV;
	   //----------------------------------------------------------------
	   u=nup+du;
	   v=nvp+dv;
	   float G=__half2float(MemG[GlobalOffset]);
	   //------------------------------------------------
	   u=u+(__half2float(MemUPrev[GlobalOffset])-u)*PrevFactor;
	   v=v+(__half2float(MemVPrev[GlobalOffset])-v)*PrevFactor;
	   //----------------------------------------------------------------
	   float Vect_lenght= sqrt(u*u+v*v);
	   if (isnan(u) || isnan(v)|| Vect_lenght<0.01 || Vect_lenght>200.01)
	   {
	       u=0.0;
		   v=0.0;
	   }
	   //----------------------------------------------------------------
	   MemU[GlobalOffset]=__float2half(u);
	   MemV[GlobalOffset]=__float2half(v);
	   //----------------------------------------------------------------
   }
   else
   {
	   if(globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
	   {
		   MemU[GlobalOffset]=__float2half(0.0f);
		   MemV[GlobalOffset]=__float2half(0.0f);

		   MemUp[GlobalOffset]=__float2half(0.0f);
	       MemVp[GlobalOffset]=__float2half(0.0f);

	       MemU0[GlobalOffset]=__float2half(0.0f);
	       MemV0[GlobalOffset]=__float2half(0.0f);
	   }
   }
}
//==========================================================================
__global__ void Iter_Compute_PEigen_Kernel(half * MemIm1,half * MemG,half * MemU,half * MemV,half * MemP1x,half *MemP2x,half *MemP1y,half *MemP2y,half * MemAPP1x,half *MemAPP2x,half *MemAPP1y,half *MemAPP2y,float Tau,float epsi,bool FirstTime,half *MNx,half *MNy,int numBlocksx,int numBlocksy,int numThreadsx,int numThreadsy,int Width,int Height)
{
	//===============================================================================================
    //
	//===============================================================================================
	int globalX = (blockIdx.x * blockDim.x + threadIdx.x)*1;

    //------------------------------------------------------------------
    if (globalX==0)
    {
    	//for (int i=0;i<2;i++)

    		//Compute_PEigen_Kernel<<<(numBlocksx,numBlocksy,1), (numThreadsx,numThreadsy,1)>>>(MemIm1,MemG, MemU,MemV, MemP1x,MemP2x,MemP1y,MemP2y,MemAPP1x,MemAPP2x,MemAPP1y,MemAPP2y,Tau,epsi,FirstTime,MNx,MNy,Width, Height);
			//cudaDeviceSynchronize();
    }

}
//==========================================================================
__global__ void OpticalFlow_Threshold_Kernel(half * MemUSrc, half * MemVSrc, half * MemUDst, half * MemVDst,float Value,bool GreatEqual, int Width, int Height)
{   //------------------------------------------------------------------
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if(globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
	{
		float fu=__half2float(MemUSrc[GlobalOffset]);
		float fv=__half2float(MemVSrc[GlobalOffset]);
		float Norm=sqrt(fu*fu+fv*fv);

		if (!GreatEqual)
		{
			if (Norm<=Value)
			{
				MemUDst[GlobalOffset]= __float2half(0.0f);
				MemVDst[GlobalOffset]= __float2half(0.0f);
			}
			else
			{
				MemUDst[GlobalOffset]= MemUSrc[GlobalOffset];
				MemVDst[GlobalOffset]= MemVSrc[GlobalOffset];
			}
		}
		else
		{
			if (Norm>=Value)
			{
				MemUDst[GlobalOffset]= __float2half(0.0f);
				MemVDst[GlobalOffset]= __float2half(0.0f);
			}
			else
			{
				MemUDst[GlobalOffset]= MemUSrc[GlobalOffset];
				MemVDst[GlobalOffset]= MemVSrc[GlobalOffset];
			}
		}

	}
}
//==========================================================================
// End Kernels
//==========================================================================
//--------------------------------------------------------------------------
TCVMotion::TCVMotion(void * d_Gpu)
{
	Gpu = d_Gpu;

	MemU=NULL;
	MemV=NULL;

	MemPyrIm1=NULL;
	MemPyrIm2=NULL;

	MemUPrev=NULL;
	MemVPrev=NULL;

    MemPyrIm2Warped=NULL;

	MemPyrG=NULL;

	MemP1=NULL;
	MemP2=NULL;
	MemP3=NULL;
	MemP4=NULL;

	MemAPP1=NULL;
	MemAPP2=NULL;
	MemAPP3=NULL;
	MemAPP4=NULL;

	MemDivx=NULL;
    MemDivy=NULL;

	MemCensus1=NULL;
	MemCensus2=NULL;

	MemIx=NULL;
	MemIy=NULL;
    MemIt=NULL;

	MemU0=NULL;
    MemV0=NULL;

	MemUp=NULL;
	MemVp=NULL;

	MemNx=NULL;
	MemNy=NULL;

	MemHFAux1=NULL;

	Scales=0;
}
//--------------------------------------------------------------------------
void TCVMotion::InitPyramid(int NumScalesMax,int MinSize,float Factor,uint Width, uint Height)
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
	      delete MemP3[i];
	      delete MemP4[i];

	      delete MemAPP1[i];
	      delete MemAPP2[i];
	      delete MemAPP3[i];
	      delete MemAPP4[i];

	      delete MemDivx[i];
	      delete MemDivy[i];

	      delete MemCensus1[i];
	      delete MemCensus2[i];

	      delete MemIx[i];
	      delete MemIy[i];
	      delete MemIt[i];

	      delete MemU[i];
	      delete MemV[i];

	      delete MemU0[i];
	      delete MemV0[i];

	      delete MemUp[i];
	      delete MemVp[i];

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
     MemP3 = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemP4 = new TGpuMem::TGpuMemHalfFloat*[NumScales];

     MemAPP1 = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemAPP2 = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemAPP3 = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemAPP4 = new TGpuMem::TGpuMemHalfFloat*[NumScales];

     MemDivx = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemDivy = new TGpuMem::TGpuMemHalfFloat*[NumScales];

     MemCensus1 = new TGpuMem::TGpuMemUInt*[NumScales];
     MemCensus2 = new TGpuMem::TGpuMemUInt*[NumScales];

     MemIx = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemIy = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemIt = new TGpuMem::TGpuMemHalfFloat*[NumScales];

     MemU = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemV = new TGpuMem::TGpuMemHalfFloat*[NumScales];

     MemU0 = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemV0 = new TGpuMem::TGpuMemHalfFloat*[NumScales];

     MemUp = new TGpuMem::TGpuMemHalfFloat*[NumScales];
     MemVp = new TGpuMem::TGpuMemHalfFloat*[NumScales];

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
		 MemP3[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemP3[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemP4[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemP4[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemAPP1[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemAPP1[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemAPP2[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemAPP2[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemAPP3[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemAPP3[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemAPP4[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemAPP4[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemDivx[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemDivx[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemDivy[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemDivy[i]->Init(0);
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
		 MemV[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemV[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemU0[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemU0[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemV0[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemV0[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemUp[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemUp[i]->Init(0);
		 //---------------------------------------------------------------------------
		 MemVp[i] = new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)Width,(uint)Height,1, false);
		 MemVp[i]->Init(0);
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

		 Width= (uint)(Width*Factor);
		 Height= (uint)(Height*Factor);

	 }

	 Scales= NumScales;
}
//--------------------------------------------------------------------------
void TCVMotion::InitPyramid()
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
			 MemP3[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemP4[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemAPP1[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemAPP2[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemAPP3[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemAPP4[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemDivx[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemDivy[i]->Init(0);
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
			 MemV[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemU0[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemV0[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemUp[i]->Init(0);
			 //---------------------------------------------------------------------------
			 MemVp[i]->Init(0);
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
void TCVMotion::GeneratePyramid(TGpuMem::TGpuMemUChar  * MemSrc,TGpuMem::TGpuMemHalfFloat  ** MemPyramid)
{
	 if (*MemPyramid!=NULL)
	 {
		 //MemSrc->Copy(MemPyramid[0]);
		 MemSrc->Casting(MemPyramid[0]);
		 for (int i=0;i<Scales-1;i++)
		 {
			 //((TGpu *)Gpu)->CV->Geometry->Resize(MemPyramid[i],MemPyramid[i+1]);
			 ((TGpu *)Gpu)->CV->Geometry->ResizeBilinear(MemPyramid[i], MemPyramid[i + 1]);
		 }
	 }
}
//--------------------------------------------------------------------------
void TCVMotion::GeneratePyramid(TGpuMem::TGpuMemHalfFloat  * MemSrc,TGpuMem::TGpuMemHalfFloat  ** MemPyramid)
{
	 if (*MemPyramid!=NULL)
	 {
		 MemSrc->Copy(MemPyramid[0]);
		 for (int i=0;i<Scales-1;i++)
		 {
			 ((TGpu *)Gpu)->CV->Geometry->Resize(MemPyramid[i],MemPyramid[i+1]);
		 }
	 }
}
//--------------------------------------------------------------------------
void TCVMotion::GeneratePyramidOF(TGpuMem::TGpuMemHalfFloat  * MemSrc,TGpuMem::TGpuMemHalfFloat  ** MemPyramid)
{
	 if (*MemPyramid!=NULL)
	 {
		 MemSrc->Copy(MemPyramid[0]);
		 for (int i=0;i<Scales-1;i++)
		 {
			 ((TGpu *)Gpu)->CV->Math->Div(MemPyramid[i],2.0f,MemHFAux2[i]);
			 ((TGpu *)Gpu)->CV->Geometry->Resize(MemHFAux2[i],MemPyramid[i+1]);
		 }
	 }
}
//--------------------------------------------------------------------------
// Standard Method

//--------------------------------------------------------------------------
void TCVMotion::AniTVL1(TGpuMem::TGpuMemHalfFloat  * MemSrc1,TGpuMem::TGpuMemHalfFloat  * MemSrc2,TGpuMem::TGpuMemHalfFloat *U,TGpuMem::TGpuMemHalfFloat *V,int NumIters,int NumWarps,float Alpha, float Beta,bool PyramidalIter)
{

	 GeneratePyramid(MemSrc1,MemPyrIm1);
	 GeneratePyramid(MemSrc2,MemPyrIm2);
	 //----------------------------------------------
     // Compute Optical Flow per each Scale
     //----------------------------------------------
	 float FactorX, FactorY;
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
        	 //((TGpu *)Gpu)->CV->Geometry->Resize(MemU[i], MemU[i - 1]);
        	 //((TGpu *)Gpu)->CV->Geometry->Resize(MemV[i], MemV[i - 1]);
			 ((TGpu *)Gpu)->CV->Geometry->ResizeBilinear(MemU[i], MemU[i - 1]);
			 ((TGpu *)Gpu)->CV->Geometry->ResizeBilinear(MemV[i], MemV[i - 1]);
             //-----------------------------------
             FactorX = (float)MemU[i - 1]->Width() / (float)MemU[i]->Width();
             FactorY = (float)MemU[i - 1]->Height() / (float)MemU[i]->Height();
             ((TGpu *)Gpu)->CV->Math->Mult(MemU[i - 1], FactorX, MemU[i - 1]);
             ((TGpu *)Gpu)->CV->Math->Mult(MemV[i - 1], FactorY, MemV[i - 1]);
             //-----------------------------------
         }
     }
     //----------------------------------------------
     MemU[0]->Copy(U);
     MemV[0]->Copy(V);

}
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
void TCVMotion::Compute_OF_TV_L1_Huber(int NumScale, int NumIters, int NumWarps, float Alpha, float Beta, bool PiramidalIteration)
{
    //---------------------------------------------------------------------------------
    bool FirstTime = true;
    int Warped = 1;
    float Theta;
    float Sigma;
    float a = 3.5f;
    float SST;
    //---------------------------------------------------------------------------------
	// Compute Census
    //---------------------------------------------------------------------------------
	((TGpu *)Gpu)->CV->Features->DiffusionWeight(MemPyrIm1[NumScale],MemPyrG[NumScale],Alpha,Beta);
	((TGpu *)Gpu)->CV->Features->Census(MemPyrIm1[NumScale], MemCensus1[NumScale],4);
    //---------------------------------------------------------------------------------
	if (MemPyrIm1[NumScale]->Width()<160 || MemPyrIm1[NumScale]->Height()<160)
	    ((TGpu *)Gpu)->CV->Features->EigenVectors(MemPyrIm1[NumScale],MemNx[NumScale],MemNy[NumScale],((TGpu *)Gpu)->CV->Filters->d_FilterGauss_1);
	else
		((TGpu *)Gpu)->CV->Features->EigenVectors(MemPyrIm1[NumScale],MemNx[NumScale],MemNy[NumScale],((TGpu *)Gpu)->CV->Filters->d_FilterGauss_2);
	//--------------------------------------------------------------------------------
	//((TGpu *)Gpu)->CV->Features->Derivates(MemPyrIm1[NumScale],MemPyrIm2[NumScale],MemIx[NumScale],MemIy[NumScale],MemIt[NumScale]);
    //--------------------------------------------------------------------------------
    if (PiramidalIteration)
    {
        float Factor = ((float)(NumScale-1 ) / (float)(Scales-1));
        NumIters = (int)(((1.0 - Factor) * NumIters) + (Factor * ((float)NumIters / 1.9f)));
        if (NumIters <= 1)
        {
            NumIters = 5;
        }
        //if (NumScale==0) NumIters = NumIters/2;
        //NumWarps = (int)(((1.0 - Factor) * NumWarps) + (Factor * ((float)NumWarps / 1.9f)));
    }

    if (NumScale==Scales-1)
    {
    	MemU[NumScale]->Init(0.0f);
    	MemV[NumScale]->Init(0.0f);

    	if (MemPyrIm1[NumScale]->Width()<=20 || MemPyrIm1[NumScale]->Height()<=20)
         	NumIters=2;
    }
    MemDivx[NumScale]->Init(0.0f);
    MemDivy[NumScale]->Init(0.0f);

    NumWarps= (int)ceil(NumIters/NumWarps)+1;
    //---------------------------------------------------------------------------------
/*
    cout<<"NumScale.."<<NumScale<<endl;
    cout<<"Scales.."<<Scales<<endl;
    cout<<"Num Iters.."<<NumIters<<endl;
    */
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
        Compute_PEigen(NumScale,FirstTime);

        /*((TGpu *)Gpu)->CV->Utils->ReplicateEdges(MemAPP1[NumScale],MemAPP1[NumScale]);
        ((TGpu *)Gpu)->CV->Utils->ReplicateEdges(MemAPP2[NumScale],MemAPP2[NumScale]);
        ((TGpu *)Gpu)->CV->Utils->ReplicateEdges(MemAPP3[NumScale],MemAPP3[NumScale]);
        ((TGpu *)Gpu)->CV->Utils->ReplicateEdges(MemAPP4[NumScale],MemAPP4[NumScale]);*/

        ((TGpu *)Gpu)->CV->Math->Divergence(MemAPP1[NumScale], MemAPP2[NumScale],MemDivx[NumScale]);
        ((TGpu *)Gpu)->CV->Math->Divergence(MemAPP3[NumScale], MemAPP4[NumScale],MemDivy[NumScale]);

       /* ((TGpu *)Gpu)->CV->Utils->ReplicateEdges(MemDivx[NumScale],MemDivx[NumScale]);
        ((TGpu *)Gpu)->CV->Utils->ReplicateEdges(MemDivy[NumScale],MemDivy[NumScale]);*/
        //-------------------------------------------------------------------
        Update_OF_Up_Vp(NumScale,FirstTime,Theta,Sigma,Warped);
        ((TGpu *)Gpu)->CV->Utils->ReplicateEdges(MemU[NumScale],MemV[NumScale]);
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
	((TGpu *)Gpu)->CV->Filters->Median3x3(MemV[NumScale],MemHFAux1[NumScale]);
	MemHFAux1[NumScale]->Copy(MemV[NumScale]);
    //---------------------------------------------------------------------------------
}
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
void TCVMotion::Update_OF_Up_Vp(int NumScale,int FirstTime,float Theta,float Sigma,int Warped)
{
    float Lambda = 125.0f;
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp((int)(MemPyrIm1[NumScale]->Width()), numThreads.x), ((TGpu *)Gpu)->iDivUp((int)(MemPyrIm1[NumScale]->Height()), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Update_OF_Up_Vp_Kernel<<<numBlocks, numThreads>>>(MemU[NumScale]->GetMemory(), MemV[NumScale]->GetMemory(),MemUp[NumScale]->GetMemory(), MemVp[NumScale]->GetMemory(),MemDivx[NumScale]->GetMemory(),MemDivy[NumScale]->GetMemory(),MemIx[NumScale]->GetMemory(),MemIy[NumScale]->GetMemory(),MemIt[NumScale]->GetMemory(),MemU0[NumScale]->GetMemory(),MemV0[NumScale]->GetMemory(),Theta, Sigma, Lambda, FirstTime, Warped,MemPyrIm1[NumScale]->Width(), MemPyrIm1[NumScale]->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVMotion::Iter_Compute_PEigen(int NumScale,bool FirstTime)
{
    float Tau = 0.5f;
    float Epsilon = 0.0001f;
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreadsIm = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocksIm =  dim3(((TGpu *)Gpu)->iDivUp(MemPyrIm1[NumScale]->Width(), numThreadsIm.x), ((TGpu *)Gpu)->iDivUp(MemPyrIm1[NumScale]->Height(), numThreadsIm.y));
	//----------------------------------------------------------------------------------------------------
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp((int)(1), numThreads.x), ((TGpu *)Gpu)->iDivUp((int)(1), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Iter_Compute_PEigen_Kernel<<<numBlocks, numThreads>>>(MemPyrIm1[NumScale]->GetMemory(),MemPyrG[NumScale]->GetMemory(), MemU[NumScale]->GetMemory(),MemV[NumScale]->GetMemory(), MemP1[NumScale]->GetMemory(),MemP2[NumScale]->GetMemory(),MemP3[NumScale]->GetMemory(),MemP4[NumScale]->GetMemory(),MemAPP1[NumScale]->GetMemory(),MemAPP2[NumScale]->GetMemory(),MemAPP3[NumScale]->GetMemory(),MemAPP4[NumScale]->GetMemory(),Tau,Epsilon,FirstTime,MemNx[NumScale]->GetMemory(),MemNy[NumScale]->GetMemory(),numThreadsIm.x,numThreadsIm.y,numThreadsIm.x,numThreadsIm.y,MemPyrIm1[NumScale]->Width(), MemPyrIm1[NumScale]->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVMotion::Compute_PEigen(int NumScale,bool FirstTime)
{
    float Tau = 0.5f;
    float Epsilon = 0.0001f;
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemPyrIm1[NumScale]->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemPyrIm1[NumScale]->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Compute_PEigen_Kernel<<<numBlocks, numThreads>>>(MemPyrIm1[NumScale]->GetMemory(),MemPyrG[NumScale]->GetMemory(), MemU[NumScale]->GetMemory(),MemV[NumScale]->GetMemory(), MemP1[NumScale]->GetMemory(),MemP2[NumScale]->GetMemory(),MemP3[NumScale]->GetMemory(),MemP4[NumScale]->GetMemory(),MemAPP1[NumScale]->GetMemory(),MemAPP2[NumScale]->GetMemory(),MemAPP3[NumScale]->GetMemory(),MemAPP4[NumScale]->GetMemory(),Tau,Epsilon,FirstTime,MemNx[NumScale]->GetMemory(),MemNy[NumScale]->GetMemory(),MemPyrIm1[NumScale]->Width(), MemPyrIm1[NumScale]->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------

// Previous OF Method

//--------------------------------------------------------------------------
void TCVMotion::AniTVL1(TGpuMem::TGpuMemHalfFloat *MemSrc1,TGpuMem::TGpuMemHalfFloat *MemSrc2,TGpuMem::TGpuMemHalfFloat *U,TGpuMem::TGpuMemHalfFloat *V,TGpuMem::TGpuMemHalfFloat *U_Prev,TGpuMem::TGpuMemHalfFloat *V_Prev,int NumIters,int NumWarps,float Alpha, float Beta,float PrevOFFactor,bool PyramidalIter)
{
	 //----------------------------------------------
	 GeneratePyramid(MemSrc1,MemPyrIm1);
	 GeneratePyramid(MemSrc2,MemPyrIm2);
	 //----------------------------------------------
	 ((TGpu*)(Gpu))->CV->Geometry->Warping(U_Prev,MemHFAux1[0],U_Prev,V_Prev,false);
	 GeneratePyramidOF(MemHFAux1[0],MemUPrev);
	 //-------------------------------------
	 ((TGpu*)(Gpu))->CV->Geometry->Warping(V_Prev,MemHFAux1[0],U_Prev,V_Prev,false);
	 GeneratePyramidOF(MemHFAux1[0],MemVPrev);
	 //----------------------------------------------
	 if (PrevOFFactor!=0.0f)
	 {
	 ((TGpu*)(Gpu))->CV->Math->Mult(MemUPrev[Scales-1],PrevOFFactor,MemU[Scales-1]);
	 ((TGpu*)(Gpu))->CV->Math->Mult(MemVPrev[Scales-1],PrevOFFactor,MemV[Scales-1]);
	 }
	 //----------------------------------------------
     // Compute Optical Flow per each Scale
     //----------------------------------------------
	 float FactorX, FactorY;
     for (int i = Scales - 1; i >= 0; i--)
     {
         //----------------------------------------------
         // Compute OF
         //----------------------------------------------
         Compute_OF_TV_L1_Huber_Prev(i,NumIters, NumWarps, Alpha, Beta, PyramidalIter,PrevOFFactor);
         //----------------------------------------------
         // Resize U,V
         //----------------------------------------------
         if (i != 0)
         {
             //-----------------------------------
        	 ((TGpu *)Gpu)->CV->Geometry->Resize(MemU[i], MemU[i - 1]);
        	 ((TGpu *)Gpu)->CV->Geometry->Resize(MemV[i], MemV[i - 1]);
             //-----------------------------------
             //-----------------------------------
             FactorX = (float)MemU[i - 1]->Width() / (float)MemU[i]->Width();
             FactorY = (float)MemU[i - 1]->Height() / (float)MemU[i]->Height();
             ((TGpu *)Gpu)->CV->Math->Mult(MemU[i - 1], FactorX, MemU[i - 1]);
             ((TGpu *)Gpu)->CV->Math->Mult(MemV[i - 1], FactorY, MemV[i - 1]);
             //-----------------------------------
             //-----------------------------------

             //-----------------------------------
         }
     }
     //----------------------------------------------
     MemU[0]->Copy(U);
     MemV[0]->Copy(V);

}
//--------------------------------------------------------------------------
void TCVMotion::Compute_OF_TV_L1_Huber_Prev(int NumScale, int NumIters, int NumWarps, float Alpha, float Beta, bool PiramidalIteration,float PrevOFFactor)
{
    //---------------------------------------------------------------------------------
    bool FirstTime = true;
    int Warped = 1;
    float Theta;
    float Sigma;
    float a = 3.5f;
    float SST;
    //---------------------------------------------------------------------------------
	// Compute Census
    //---------------------------------------------------------------------------------
	((TGpu *)Gpu)->CV->Features->DiffusionWeight(MemPyrIm1[NumScale],MemPyrG[NumScale],Alpha,Beta);
	((TGpu *)Gpu)->CV->Features->Census(MemPyrIm1[NumScale], MemCensus1[NumScale],4);
    //---------------------------------------------------------------------------------
	if (MemPyrIm1[NumScale]->Width()<160 || MemPyrIm1[NumScale]->Height()<160)
	    ((TGpu *)Gpu)->CV->Features->EigenVectors(MemPyrIm1[NumScale],MemNx[NumScale],MemNy[NumScale],((TGpu *)Gpu)->CV->Filters->d_FilterGauss_1);
	else
		((TGpu *)Gpu)->CV->Features->EigenVectors(MemPyrIm1[NumScale],MemNx[NumScale],MemNy[NumScale],((TGpu *)Gpu)->CV->Filters->d_FilterGauss_2);
	//--------------------------------------------------------------------------------
	//((TGpu *)Gpu)->CV->Features->Derivates(MemPyrIm1[NumScale],MemPyrIm2[NumScale],MemIx[NumScale],MemIy[NumScale],MemIt[NumScale]);
    //--------------------------------------------------------------------------------
    if (true)
    {
        float Factor = ((float)(NumScale-1 ) / (float)(Scales-1));
        NumIters = (int)(((1.0 - Factor) * NumIters) + (Factor * ((float)NumIters / 1.9f)));
        if (NumIters <= 1)
        {
            NumIters = 5;
        }
        //if (NumScale==0) NumIters = NumIters/2;
        //NumWarps = (int)(((1.0 - Factor) * NumWarps) + (Factor * ((float)NumWarps / 1.9f)));
    }
    if (NumScale==Scales-1)
    {
    	MemU[NumScale]->Init(0.0f);
    	MemV[NumScale]->Init(0.0f);
    	if (MemPyrIm1[NumScale]->Width()<=20 || MemPyrIm1[NumScale]->Height()<=20)
         	NumIters=2;
    }
    MemDivx[NumScale]->Init(0.0f);
    MemDivy[NumScale]->Init(0.0f);

    NumWarps= (int)ceil(NumIters/NumWarps)+1;
    //---------------------------------------------------------------------------------
/*
    cout<<"NumScale.."<<NumScale<<endl;
    cout<<"Scales.."<<Scales<<endl;
    cout<<"Num Iters.."<<NumIters<<endl;
    */
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
        Compute_PEigen(NumScale,FirstTime);

        ((TGpu *)Gpu)->CV->Math->Divergence(MemAPP1[NumScale], MemAPP2[NumScale],MemDivx[NumScale]);
        ((TGpu *)Gpu)->CV->Math->Divergence(MemAPP3[NumScale], MemAPP4[NumScale],MemDivy[NumScale]);

        Update_OF_Up_Vp_Prev(NumScale,FirstTime,Theta,Sigma,Warped,PrevOFFactor);

        ((TGpu *)Gpu)->CV->Utils->ReplicateEdges(MemU[NumScale],MemV[NumScale]);

        FirstTime=false;
        Warped=0;

    }
    //---------------------------------------------------------------------------------
	// Median Filter
    //---------------------------------------------------------------------------------
	((TGpu *)Gpu)->CV->Filters->Median3x3(MemU[NumScale],MemHFAux1[NumScale]);
	MemHFAux1[NumScale]->Copy(MemU[NumScale]);
	((TGpu *)Gpu)->CV->Filters->Median3x3(MemV[NumScale],MemHFAux1[NumScale]);
	MemHFAux1[NumScale]->Copy(MemV[NumScale]);
    //---------------------------------------------------------------------------------
}
//--------------------------------------------------------------------------
void TCVMotion::Update_OF_Up_Vp_Prev(int NumScale,int FirstTime,float Theta,float Sigma,int Warped,float PrevOFFactor)
{
    float Lambda = 125.0f;
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp((int)(MemPyrIm1[NumScale]->Width()), numThreads.x), ((TGpu *)Gpu)->iDivUp((int)(MemPyrIm1[NumScale]->Height()), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Update_OF_Up_Vp_Prev_Kernel<<<numBlocks, numThreads>>>(MemU[NumScale]->GetMemory(), MemV[NumScale]->GetMemory(),MemPyrG[NumScale]->GetMemory(),MemUPrev[NumScale]->GetMemory(), MemVPrev[NumScale]->GetMemory(), MemUp[NumScale]->GetMemory(), MemVp[NumScale]->GetMemory(),MemDivx[NumScale]->GetMemory(),MemDivy[NumScale]->GetMemory(),MemIx[NumScale]->GetMemory(),MemIy[NumScale]->GetMemory(),MemIt[NumScale]->GetMemory(),MemU0[NumScale]->GetMemory(),MemV0[NumScale]->GetMemory(),Theta, Sigma, Lambda, FirstTime,PrevOFFactor, Warped,MemPyrIm1[NumScale]->Width(), MemPyrIm1[NumScale]->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVMotion::Compute_Warping(int NumScale)
{
    //---------------------------------------------------------------------------------
	// Median Filter
    //---------------------------------------------------------------------------------
	((TGpu *)Gpu)->CV->Filters->Median3x3(MemU[NumScale],MemHFAux1[NumScale]);
	MemHFAux1[NumScale]->Copy(MemU[NumScale]);
	((TGpu *)Gpu)->CV->Filters->Median3x3(MemV[NumScale],MemHFAux1[NumScale]);
	MemHFAux1[NumScale]->Copy(MemV[NumScale]);
    //---------------------------------------------------------------------------------
	((TGpu *)Gpu)->CV->Geometry->Warping(MemPyrIm2[NumScale],MemPyrIm2Warped[NumScale],MemU[NumScale],MemV[NumScale],true);
	((TGpu *)Gpu)->CV->Features->Census(MemPyrIm2Warped[NumScale], MemCensus2[NumScale],4);
	((TGpu *)Gpu)->CV->Filters->CensusDerivates(MemCensus1[NumScale],MemCensus2[NumScale],MemIx[NumScale],MemIy[NumScale],MemIt[NumScale]);
    //---------------------------------------------------------------------------------
}
//--------------------------------------------------------------------------
void TCVMotion::Compute_Census_Derivates(int NumScale, TGpuMem::TGpuMemHalfFloat  * _MemIm1,TGpuMem::TGpuMemHalfFloat  * _MemIm2, TGpuMem::TGpuMemHalfFloat *_MemIx, TGpuMem::TGpuMemHalfFloat *_MemIy, TGpuMem::TGpuMemHalfFloat *_MemIt)
{
	TGpuMem::TGpuMemUInt *_MemCensus1=new TGpuMem::TGpuMemUInt(Gpu,(uint)_MemIm1->Width(),(uint)_MemIm1->Height(),1, false);
	TGpuMem::TGpuMemUInt *_MemCensus2=new TGpuMem::TGpuMemUInt(Gpu,(uint)_MemIm1->Width(),(uint)_MemIm1->Height(),1, false);

	TGpuMem::TGpuMemUInt *_MemCensus1Aux=new TGpuMem::TGpuMemUInt(Gpu,(uint)_MemIm1->Width(),(uint)_MemIm1->Height(),1, false);
	TGpuMem::TGpuMemUInt *_MemCensus2Aux=new TGpuMem::TGpuMemUInt(Gpu,(uint)_MemIm1->Width(),(uint)_MemIm1->Height(),1, false);

	TGpuMem::TGpuMemHalfFloat *_UW=new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)_MemIm1->Width(),(uint)_MemIm1->Height(),1, false);
	TGpuMem::TGpuMemHalfFloat *_VW=new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)_MemIm1->Width(),(uint)_MemIm1->Height(),1, false);
	TGpuMem::TGpuMemHalfFloat *_MemAux1=new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)_MemIm1->Width(),(uint)_MemIm1->Height(),1, false);
	TGpuMem::TGpuMemHalfFloat *_MemAux2=new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)_MemIm1->Width(),(uint)_MemIm1->Height(),1, false);

	_MemAux1->Init(0.0f);
	_MemAux2->Init(0.0f);

	((TGpu *)Gpu)->CV->Geometry->Warping(MemPyrIm2[NumScale],MemPyrIm2Warped[NumScale],MemU[NumScale],MemV[NumScale],true);

	((TGpu *)Gpu)->CV->Features->Census(MemPyrIm1[NumScale], _MemCensus1,4);
	((TGpu *)Gpu)->CV->Features->Census(MemPyrIm2Warped[NumScale], _MemCensus2,4);

	((TGpu *)Gpu)->CV->Math->HammingDistance(_MemCensus1,_MemCensus2,MemIt[NumScale],1.0/4.0);
	//--------------------------------------------------------------------------
	// Ix
	//--------------------------------------------------------------------------
	//_UW->Init(0.5f);
	//_VW->Init(0.0f);

	_UW->Init(1.0f);
	_VW->Init(0.0f);

	((TGpu *)Gpu)->CV->Geometry->Warping(MemPyrIm2Warped[NumScale],_MemAux1,_UW,_VW,true);
	//--------------------------------------------------------------------------
	//_UW->Init(-0.5f);
	//_VW->Init(0.0f);

	_UW->Init(-1.0f);
	_VW->Init(0.0f);
	((TGpu *)Gpu)->CV->Geometry->Warping(MemPyrIm2Warped[NumScale],_MemAux2,_UW,_VW,true);
	//--------------------------------------------------------------------------
	((TGpu *)Gpu)->CV->Features->Census(_MemAux1, _MemCensus1Aux,4);
	((TGpu *)Gpu)->CV->Features->Census(_MemAux2, _MemCensus2Aux,4);
	((TGpu *)Gpu)->CV->Math->HammingDistance(_MemCensus1Aux,_MemCensus1,_UW,1.0f);
	((TGpu *)Gpu)->CV->Math->HammingDistance(_MemCensus2Aux,_MemCensus1,_VW,1.0f);
	((TGpu *)Gpu)->CV->Math->Subtract(_UW,_VW,MemIx[NumScale]);
	//--------------------------------------------------------------------------
	// Iy
	//--------------------------------------------------------------------------
	//_UW->Init(0.0f);
	//_VW->Init(0.5f);

	_UW->Init(0.0f);
	_VW->Init(1.0f);
	((TGpu *)Gpu)->CV->Geometry->Warping(MemPyrIm2Warped[NumScale],_MemAux1,_UW,_VW,true);
	//--------------------------------------------------------------------------
	//_UW->Init(0.0f);
	//_VW->Init(-0.5f);

	_UW->Init(0.0f);
	_VW->Init(-1.0f);
	((TGpu *)Gpu)->CV->Geometry->Warping(MemPyrIm2Warped[NumScale],_MemAux2,_UW,_VW,true);
	//--------------------------------------------------------------------------
	((TGpu *)Gpu)->CV->Features->Census(_MemAux1, _MemCensus1Aux,4);
	((TGpu *)Gpu)->CV->Features->Census(_MemAux2, _MemCensus2Aux,4);
	((TGpu *)Gpu)->CV->Math->HammingDistance(_MemCensus1Aux,_MemCensus1,_UW,1.0f);
	((TGpu *)Gpu)->CV->Math->HammingDistance(_MemCensus2Aux,_MemCensus1,_VW,1.0f);
	((TGpu *)Gpu)->CV->Math->Subtract(_UW,_VW,MemIy[NumScale]);
	//--------------------------------------------------------------------------

	delete _MemCensus1;
	delete _MemCensus2;
	delete _MemCensus1Aux;
	delete _MemCensus2Aux;
	delete _UW;
	delete _VW;
	delete _MemAux1;
	delete _MemAux2;
}
//--------------------------------------------------------------------------
void TCVMotion::OpticalFlow_Threshold(TGpuMem::TGpuMemHalfFloat * MemUSrc,TGpuMem::TGpuMemHalfFloat * MemVSrc,TGpuMem::TGpuMemHalfFloat * MemUDst,TGpuMem::TGpuMemHalfFloat * MemVDst,float Value,bool GreatEqual)
{
	   //----------------------------------------------------------------------------------------------------
	   // Estimate the number of Blocks and number Threads
	   //----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemUSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemUSrc->Height(), numThreads.y));
	   //----------------------------------------------------------------------------------------------------
    OpticalFlow_Threshold_Kernel<<<numBlocks, numThreads>>>(MemUSrc->GetMemory(),MemVSrc->GetMemory(), MemUDst->GetMemory(),MemVDst->GetMemory(), Value,GreatEqual,MemUSrc->Width(),MemUSrc->Height());
    cudaThreadSynchronize();

}
//--------------------------------------------------------------------------
TCVMotion::~TCVMotion()
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
	      delete MemP3[i];
	      delete MemP4[i];

	      delete MemAPP1[i];
	      delete MemAPP2[i];
	      delete MemAPP3[i];
	      delete MemAPP4[i];

	      delete MemDivx[i];
	      delete MemDivy[i];

	      delete MemCensus1[i];
	      delete MemCensus2[i];

	      delete MemIx[i];
	      delete MemIy[i];
	      delete MemIt[i];

	      delete MemU[i];
	      delete MemV[i];

	      delete MemU0[i];
	      delete MemV0[i];

	      delete MemUp[i];
	      delete MemVp[i];

	      delete MemNx[i];
	      delete MemNy[i];

	      delete MemHFAux1[i];
	      delete MemHFAux2[i];
	      delete MemHFAux3[i];
    }
}
//--------------------------------------------------------------------------


