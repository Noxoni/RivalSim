"""Pinned Bullet/ODE box-box detector for the fixed Octane child boxes."""

import warp as wp

_BULLET_OCTANE_BOX_BOX = r"""
    // Direct bounded translation of btBoxBoxDetector.cpp::dBoxBox2,
    // intersectRectQuad2, cullPoints2, and dLineClosestApproach from the
    // pinned RocketSim Bullet subtree. The two boxes have the same fixed
    // Octane child shape and translation; maxc is exactly four.
    auto add=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a,b);
    #else
        volatile float v=a+b;return v;
    #endif
    };
    auto sub=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a,b);
    #else
        volatile float v=a-b;return v;
    #endif
    };
    auto mul=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a,b);
    #else
        volatile float v=a*b;return v;
    #endif
    };
    auto div=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a,b);
    #else
        volatile float v=a/b;return v;
    #endif
    };
    struct V3{float x,y,z;};
    auto v3=[](float x,float y,float z)->V3{V3 v={x,y,z};return v;};
    auto vadd=[&](V3 a,V3 b)->V3{return v3(add(a.x,b.x),add(a.y,b.y),add(a.z,b.z));};
    auto vsub=[&](V3 a,V3 b)->V3{return v3(sub(a.x,b.x),sub(a.y,b.y),sub(a.z,b.z));};
    auto vscale=[&](V3 a,float s)->V3{return v3(mul(a.x,s),mul(a.y,s),mul(a.z,s));};
    auto vdot=[&](V3 a,V3 b)->float{return add(add(mul(a.x,b.x),mul(a.y,b.y)),mul(a.z,b.z));};
    auto component=[](V3 a,int i)->float{return i==0?a.x:(i==1?a.y:a.z);};
    auto set_component=[](V3& a,int i,float v){if(i==0)a.x=v;else if(i==1)a.y=v;else a.z=v;};
    auto column=[&](const wp::mat_t<3,3,wp::float32>& m,int c)->V3{
        return v3(m.data[0][c],m.data[1][c],m.data[2][c]);
    };
    auto matrix_vector=[&](const wp::mat_t<3,3,wp::float32>& m,V3 v)->V3{
        return v3(
            add(add(mul(m.data[0][0],v.x),mul(m.data[0][1],v.y)),mul(m.data[0][2],v.z)),
            add(add(mul(m.data[1][0],v.x),mul(m.data[1][1],v.y)),mul(m.data[1][2],v.z)),
            add(add(mul(m.data[2][0],v.x),mul(m.data[2][1],v.y)),mul(m.data[2][2],v.z)));
    };
    auto transpose_vector=[&](const wp::mat_t<3,3,wp::float32>& m,V3 v)->V3{
        return v3(vdot(column(m,0),v),vdot(column(m,1),v),vdot(column(m,2),v));
    };

    const V3 root_a=v3(position_a[0],position_a[1],position_a[2]);
    const V3 root_b=v3(position_b[0],position_b[1],position_b[2]);
    const V3 child_offset=v3(0.277513981f,0.0f,0.415099978f);
    const V3 p1=vadd(root_a,matrix_vector(basis_a,child_offset));
    const V3 p2=vadd(root_b,matrix_vector(basis_b,child_offset));
    const float A[3]={1.20372915f,0.865653098f,0.385250092f};
    const float B[3]={1.20372915f,0.865653098f,0.385250092f};
    const V3 p=vsub(p2,p1);
    const V3 pp=transpose_vector(basis_a,p);
    float R[3][3],Q[3][3];
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){
        R[i][j]=vdot(column(basis_a,i),column(basis_b,j));
        Q[i][j]=fabsf(R[i][j]);
    }

    float s=-3.402823466e+38f;
    int invert=0;
    int code=0;
    int normal_kind=0;
    int normal_axis=0;
    V3 normal_c=v3(0.0f,0.0f,0.0f);
    auto face_test=[&](float expression,float radius,int kind,int axis,int candidate)->bool{
        const float s2=sub(fabsf(expression),radius);
        if(s2>0.0f)return false;
        if(s2>s){s=s2;normal_kind=kind;normal_axis=axis;invert=expression<0.0f;code=candidate;}
        return true;
    };
    if(!face_test(pp.x,add(add(add(A[0],mul(B[0],Q[0][0])),mul(B[1],Q[0][1])),mul(B[2],Q[0][2])),1,0,1))return;
    if(!face_test(pp.y,add(add(add(A[1],mul(B[0],Q[1][0])),mul(B[1],Q[1][1])),mul(B[2],Q[1][2])),1,1,2))return;
    if(!face_test(pp.z,add(add(add(A[2],mul(B[0],Q[2][0])),mul(B[1],Q[2][1])),mul(B[2],Q[2][2])),1,2,3))return;
    const float bp0=vdot(column(basis_b,0),p);
    const float bp1=vdot(column(basis_b,1),p);
    const float bp2=vdot(column(basis_b,2),p);
    if(!face_test(bp0,add(add(add(mul(A[0],Q[0][0]),mul(A[1],Q[1][0])),mul(A[2],Q[2][0])),B[0]),2,0,4))return;
    if(!face_test(bp1,add(add(add(mul(A[0],Q[0][1]),mul(A[1],Q[1][1])),mul(A[2],Q[2][1])),B[1]),2,1,5))return;
    if(!face_test(bp2,add(add(add(mul(A[0],Q[0][2]),mul(A[1],Q[1][2])),mul(A[2],Q[2][2])),B[2]),2,2,6))return;

    for(int i=0;i<3;++i)for(int j=0;j<3;++j)Q[i][j]=add(Q[i][j],1.0e-5f);
    auto edge_test=[&](float expression,float radius,V3 axis,int candidate)->bool{
        float s2=sub(fabsf(expression),radius);
        if(s2>1.1920928955078125e-7f)return false;
        const float length=sqrtf(add(add(mul(axis.x,axis.x),mul(axis.y,axis.y)),mul(axis.z,axis.z)));
        if(length>1.1920928955078125e-7f){
            s2=div(s2,length);
            if(mul(s2,1.05f)>s){
                // dBoxBox2 divides every normal component by ``l``. Forming
                // one reciprocal and multiplying flips source-reachable edge
                // signs when a later dDOT14 is nearly zero.
                s=s2;normal_kind=0;normal_c=v3(
                    div(axis.x,length),div(axis.y,length),div(axis.z,length));
                invert=expression<0.0f;code=candidate;
            }
        }
        return true;
    };
    if(!edge_test(sub(mul(pp.z,R[1][0]),mul(pp.y,R[2][0])),add(add(add(mul(A[1],Q[2][0]),mul(A[2],Q[1][0])),mul(B[1],Q[0][2])),mul(B[2],Q[0][1])),v3(0.0f,-R[2][0],R[1][0]),7))return;
    if(!edge_test(sub(mul(pp.z,R[1][1]),mul(pp.y,R[2][1])),add(add(add(mul(A[1],Q[2][1]),mul(A[2],Q[1][1])),mul(B[0],Q[0][2])),mul(B[2],Q[0][0])),v3(0.0f,-R[2][1],R[1][1]),8))return;
    if(!edge_test(sub(mul(pp.z,R[1][2]),mul(pp.y,R[2][2])),add(add(add(mul(A[1],Q[2][2]),mul(A[2],Q[1][2])),mul(B[0],Q[0][1])),mul(B[1],Q[0][0])),v3(0.0f,-R[2][2],R[1][2]),9))return;
    if(!edge_test(sub(mul(pp.x,R[2][0]),mul(pp.z,R[0][0])),add(add(add(mul(A[0],Q[2][0]),mul(A[2],Q[0][0])),mul(B[1],Q[1][2])),mul(B[2],Q[1][1])),v3(R[2][0],0.0f,-R[0][0]),10))return;
    if(!edge_test(sub(mul(pp.x,R[2][1]),mul(pp.z,R[0][1])),add(add(add(mul(A[0],Q[2][1]),mul(A[2],Q[0][1])),mul(B[0],Q[1][2])),mul(B[2],Q[1][0])),v3(R[2][1],0.0f,-R[0][1]),11))return;
    if(!edge_test(sub(mul(pp.x,R[2][2]),mul(pp.z,R[0][2])),add(add(add(mul(A[0],Q[2][2]),mul(A[2],Q[0][2])),mul(B[0],Q[1][1])),mul(B[1],Q[1][0])),v3(R[2][2],0.0f,-R[0][2]),12))return;
    if(!edge_test(sub(mul(pp.y,R[0][0]),mul(pp.x,R[1][0])),add(add(add(mul(A[0],Q[1][0]),mul(A[1],Q[0][0])),mul(B[1],Q[2][2])),mul(B[2],Q[2][1])),v3(-R[1][0],R[0][0],0.0f),13))return;
    if(!edge_test(sub(mul(pp.y,R[0][1]),mul(pp.x,R[1][1])),add(add(add(mul(A[0],Q[1][1]),mul(A[1],Q[0][1])),mul(B[0],Q[2][2])),mul(B[2],Q[2][0])),v3(-R[1][1],R[0][1],0.0f),14))return;
    if(!edge_test(sub(mul(pp.y,R[0][2]),mul(pp.x,R[1][2])),add(add(add(mul(A[0],Q[1][2]),mul(A[1],Q[0][2])),mul(B[0],Q[2][1])),mul(B[1],Q[2][0])),v3(-R[1][2],R[0][2],0.0f),15))return;
    if(code==0)return;

    V3 normal=normal_kind==1?column(basis_a,normal_axis):(normal_kind==2?column(basis_b,normal_axis):matrix_vector(basis_a,normal_c));
    if(invert)normal=vscale(normal,-1.0f);
    const float depth=-s;
    const V3 normal_b=vscale(normal,-1.0f);
    normal_on_b=wp::vec_t<3,wp::float32>(normal_b.x,normal_b.y,normal_b.z);
    return_code=code;
    auto emit=[&](V3 point_b,float contact_depth){
        const float distance=-contact_depth;
        const wp::vec_t<3,wp::float32> point(point_b.x,point_b.y,point_b.z);
        if(contact_count==0){point_b0=point;distance0=distance;}
        else if(contact_count==1){point_b1=point;distance1=distance;}
        else if(contact_count==2){point_b2=point;distance2=distance;}
        else if(contact_count==3){point_b3=point;distance3=distance;}
        ++contact_count;
    };

    if(code>6){
        V3 pa=p1,pb=p2;
        for(int j=0;j<3;++j){
            const V3 ca=column(basis_a,j),cb=column(basis_b,j);
            const float sign_a=vdot(normal,ca)>0.0f?1.0f:-1.0f;
            const float sign_b=vdot(normal,cb)>0.0f?-1.0f:1.0f;
            pa=vadd(pa,vscale(ca,mul(sign_a,A[j])));
            pb=vadd(pb,vscale(cb,mul(sign_b,B[j])));
        }
        const V3 ua=column(basis_a,(code-7)/3);
        const V3 ub=column(basis_b,(code-7)%3);
        const V3 delta=vsub(pb,pa);
        const float uaub=vdot(ua,ub);
        const float q1=vdot(ua,delta);
        const float q2=-vdot(ub,delta);
        float denominator=sub(1.0f,mul(uaub,uaub));
        float alpha=0.0f,beta=0.0f;
        if(denominator>0.0001f){
            denominator=div(1.0f,denominator);
            alpha=mul(add(q1,mul(uaub,q2)),denominator);
            beta=mul(add(mul(uaub,q1),q2),denominator);
        }
        pa=vadd(pa,vscale(ua,alpha));
        pb=vadd(pb,vscale(ub,beta));
        emit(pb,depth);
        return;
    }

    const bool reference_a=code<=3;
    const wp::mat_t<3,3,wp::float32>& Ra=reference_a?basis_a:basis_b;
    const wp::mat_t<3,3,wp::float32>& Rb=reference_a?basis_b:basis_a;
    const V3 pa=reference_a?p1:p2;
    const V3 pb=reference_a?p2:p1;
    const float* Sa=reference_a?A:B;
    const float* Sb=reference_a?B:A;
    const V3 normal2=reference_a?normal:vscale(normal,-1.0f);
    const V3 nr=transpose_vector(Rb,normal2);
    const float anr[3]={fabsf(nr.x),fabsf(nr.y),fabsf(nr.z)};
    int lanr,a1,a2;
    if(anr[1]>anr[0]){
        if(anr[1]>anr[2]){a1=0;lanr=1;a2=2;}
        else{a1=0;a2=1;lanr=2;}
    }else{
        if(anr[0]>anr[2]){lanr=0;a1=1;a2=2;}
        else{a1=0;a2=1;lanr=2;}
    }
    const V3 rb_lanr=column(Rb,lanr);
    V3 center=vsub(pb,pa);
    center=vadd(center,vscale(rb_lanr,component(nr,lanr)<0.0f?Sb[lanr]:-Sb[lanr]));
    const int code_n=reference_a?code-1:code-4;
    int code1,code2;
    if(code_n==0){code1=1;code2=2;}
    else if(code_n==1){code1=0;code2=2;}
    else{code1=0;code2=1;}
    const V3 ra1=column(Ra,code1),ra2=column(Ra,code2);
    const V3 rb1=column(Rb,a1),rb2=column(Rb,a2);
    const float c1=vdot(center,ra1),c2=vdot(center,ra2);
    float m11=vdot(ra1,rb1),m12=vdot(ra1,rb2),m21=vdot(ra2,rb1),m22=vdot(ra2,rb2);
    const float k1=mul(m11,Sb[a1]),k2=mul(m21,Sb[a1]);
    const float k3=mul(m12,Sb[a2]),k4=mul(m22,Sb[a2]);
    float q[16]={
        sub(sub(c1,k1),k3),sub(sub(c2,k2),k4),
        add(sub(c1,k1),k3),add(sub(c2,k2),k4),
        add(add(c1,k1),k3),add(add(c2,k2),k4),
        sub(add(c1,k1),k3),sub(add(c2,k2),k4)};
    float ret[16]={0},buffer[16]={0};
    float* source=q;
    float* target=ret;
    int nq=4,nr_count=0;
    for(int direction=0;direction<=1;++direction){
        for(int sign=-1;sign<=1;sign+=2){
            nr_count=0;
            for(int index=0;index<nq;++index){
                float* current=source+index*2;
                float* next=source+((index+1<nq)?(index+1)*2:0);
                const bool inside=mul(static_cast<float>(sign),current[direction])<(direction==0?Sa[code1]:Sa[code2]);
                const bool next_inside=mul(static_cast<float>(sign),next[direction])<(direction==0?Sa[code1]:Sa[code2]);
                if(inside){target[nr_count*2]=current[0];target[nr_count*2+1]=current[1];++nr_count;}
                if(inside!=next_inside){
                    const int other=1-direction;
                    const float h=direction==0?Sa[code1]:Sa[code2];
                    const float slope=div(sub(next[other],current[other]),sub(next[direction],current[direction]));
                    const float boundary_delta=sub(mul(static_cast<float>(sign),h),current[direction]);
                    target[nr_count*2+other]=add(current[other],mul(slope,boundary_delta));
                    target[nr_count*2+direction]=mul(static_cast<float>(sign),h);
                    ++nr_count;
                }
                if(nr_count>=8)break;
            }
            source=target;
            target=source==ret?buffer:ret;
            nq=nr_count;
        }
    }
    if(source!=ret)for(int i=0;i<nr_count*2;++i)ret[i]=source[i];
    const int n=nr_count;
    if(n<1)return;
    const float det=div(1.0f,sub(mul(m11,m22),mul(m12,m21)));
    m11=mul(m11,det);m12=mul(m12,det);m21=mul(m21,det);m22=mul(m22,det);
    V3 points[8];float dep[8];int cnum=0;
    for(int j=0;j<n;++j){
        const float dx=sub(ret[j*2],c1),dy=sub(ret[j*2+1],c2);
        const float face_k1=sub(mul(m22,dx),mul(m12,dy));
        const float face_k2=add(mul(-m21,dx),mul(m11,dy));
        points[cnum]=vadd(vadd(center,vscale(rb1,face_k1)),vscale(rb2,face_k2));
        dep[cnum]=sub(Sa[code_n],vdot(normal2,points[cnum]));
        if(dep[cnum]>=0.0f){ret[cnum*2]=ret[j*2];ret[cnum*2+1]=ret[j*2+1];++cnum;}
    }
    if(cnum<1)return;
    int selected[4]={0,0,0,0};
    int output_count=cnum<4?cnum:4;
    if(cnum<=4){for(int i=0;i<cnum;++i)selected[i]=i;}
    else{
        int deepest=0;float max_depth=dep[0];
        for(int i=1;i<cnum;++i)if(dep[i]>max_depth){max_depth=dep[i];deepest=i;}
        float cx=0.0f,cy=0.0f,area=0.0f;
        for(int i=0;i<cnum-1;++i){
            const float amount=sub(mul(ret[i*2],ret[i*2+3]),mul(ret[i*2+2],ret[i*2+1]));
            area=add(area,amount);
            cx=add(cx,mul(amount,add(ret[i*2],ret[i*2+2])));
            cy=add(cy,mul(amount,add(ret[i*2+1],ret[i*2+3])));
        }
        const float last=sub(mul(ret[cnum*2-2],ret[1]),mul(ret[0],ret[cnum*2-1]));
        const float total=add(area,last);
        const float centroid_factor=fabsf(total)>1.1920928955078125e-7f?div(1.0f,mul(3.0f,total)):1.0e30f;
        cx=mul(centroid_factor,add(cx,mul(last,add(ret[cnum*2-2],ret[0]))));
        cy=mul(centroid_factor,add(cy,mul(last,add(ret[cnum*2-1],ret[1]))));
        float angles[8];int available[8];
        for(int i=0;i<cnum;++i){
        #if defined(__CUDA_ARCH__)
            angles[i]=static_cast<float>(atan2(double(sub(ret[i*2+1],cy)),double(sub(ret[i*2],cx))));
        #else
            angles[i]=atan2f(sub(ret[i*2+1],cy),sub(ret[i*2],cx));
        #endif
            available[i]=1;
        }
        available[deepest]=0;selected[0]=deepest;
        for(int j=1;j<4;++j){
            float desired=add(mul(static_cast<float>(j),div(mul(2.0f,3.14159265f),4.0f)),angles[deepest]);
            if(desired>3.14159265f)desired=sub(desired,mul(2.0f,3.14159265f));
            float max_diff=1.0e9f;int choice=deepest;
            for(int i=0;i<cnum;++i)if(available[i]){
                float difference=fabsf(sub(angles[i],desired));
                if(difference>3.14159265f)difference=sub(mul(2.0f,3.14159265f),difference);
                if(difference<max_diff){max_diff=difference;choice=i;}
            }
            selected[j]=choice;available[choice]=0;
        }
    }
    for(int j=0;j<output_count;++j){
        const int i=selected[j];
        V3 world=vadd(points[i],pa);
        if(!reference_a)world=vsub(world,vscale(normal,dep[i]));
        emit(world,dep[i]);
    }
"""


@wp.func_native(_BULLET_OCTANE_BOX_BOX)
def bullet_octane_box_box(
    position_a: wp.vec3,
    basis_a: wp.mat33,
    position_b: wp.vec3,
    basis_b: wp.mat33,
    normal_on_b: wp.ref[wp.vec3],
    point_b0: wp.ref[wp.vec3],
    point_b1: wp.ref[wp.vec3],
    point_b2: wp.ref[wp.vec3],
    point_b3: wp.ref[wp.vec3],
    distance0: wp.ref[wp.float32],
    distance1: wp.ref[wp.float32],
    distance2: wp.ref[wp.float32],
    distance3: wp.ref[wp.float32],
    contact_count: wp.ref[wp.int32],
    return_code: wp.ref[wp.int32],
): ...


__all__ = ["bullet_octane_box_box"]
