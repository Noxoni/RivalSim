"""Generated AMD RSQRTSS estimate table for the pinned v0.3 authority host."""

from __future__ import annotations

import base64
import hashlib
import zlib
from functools import lru_cache

import numpy as np

AMD_RSQRTSS_TABLE_SHA256 = "72A1E9C63AE43AAF0F3181C458533A8ACEF3BB91C58AF260C5C957B0C1FC8E90"
_AMD_RSQRTSS_TABLE_B85 = (
    b"c-l?d1Ggm05{BV9XLj|T-"
    b"Bp=6wr$(CZQHhO+qP}nwr!jD?Krvans|OgWkp7O>6`1DgW0}WnCY8=>Aq>0>YIYezDd4`nBW_aalWw_;~R}pzL6N=8}1v1p}"
    b"rv)>>GrEz5(d(>xaI+-o9Sw>Fa^+zHaF1>w?a{PUz_C;A@X|zP4!NYmHXEmT2K?j%H|zCcegK<ZFlqzWS)=tBX3m+P+$-"
    b">8pY2zG|rItAfhDN~q|o;46=EzOpFeD~(dVk|^OTj$*!|z9J~>D};i+0?6;nhrGT#$nDGJ%ZVJm?8xTJiY&g&$mGk23`mc3z"
    b"O+c=ON~^%lt|%Aj%2>1z9dNOON4~J1c>j8hq%5ti0zBzi-{P%=!oWviYUIwh~$fi2ndgGzOXPp4dwIT`uqs-"
    b"h4BT$=W`I`3ypt${~(m}2fy(PKb;@=?tH^n=L<eNpYYN7fcMTjyu};mHC{O{@xpnIXL#y7!DBpf9^!#>ANQQQxP#lyE!@Nn="
    b"Q^%AS8>I;jElJ7oX0twb<W_la|$P&6F81z&QTn}VdoGIItQ@d*@wN@<Lt&R>~wZuyR!{joh{gmP0mJaz<OsL);epj+F6B_Sm"
    b"7+kGAwnLV6n3Z3!Md+k9p2q%)xAD7G^p#Fx{DEO?9SVGA20_F~J#+an4wb!Dwd`Mq-"
    b"3B9K)QU7~%}ZAPjT{pg;OKebLA1jb2Vq^gwr~8@i&4(;1zdj_BaDx7s;v(FU!ZR%q$8Ky#-Vnxcu*7>&@-"
    b"X@L4pJ=AsTShby6sEHa*byRbzqKZ=)l~B>CfbuAZvQ8P4Mk$m)ai<uHq6i8*g-"
    b"{R$kRSP+yvT#x$mQfj4rE6*WOcG2GcqBglL6_G4r!3uNrjY1f#gmyBt;S=Mj|I65+FX}IdKsOu@MU~ofwFYXo%`WL1aWiL_~"
    b"1HBOJoQI2sBbT=<<3gh4QTPN0Gi8vo)SCzKN^=nsD5SI|%Vz;}GZ*Pt)>j8FI&^a1bj4sY=$=rvy9C0+zQ$1^;|6Fd%jgok*"
    b"5`$6|`7k6+Qw}Nis2Cm~;&{bT)Wn99=pbI#Ub2uAx2B&cfCvhU^IF8{cjszXXAsoa3><`+9z1V}@*oB?gA+`r?!&Yp;W^BSn"
    b"YzSJ9bwO*f2CK0OE3pF0gO*`w&=M@hA}quL%*Q-2H)sxKV-{v&2Bu?L&{Rwbnv6-"
    b"9hzS^vaTqJc1dYZhjKl~G$1n^H8iK(=gD?;S&>#KK7kxzUpkC;S9_Ws4=!!0)b5JLAL<h7-JG4ccpw?&=)DkVw9L>-"
    b"aP0(003TlW3sE>N6i#n(+Y6aCq4OB-pR7Dk34yuHTK^0IQ<xm!7P#UE~$)FM_j$$Z^A}EYPqF_(~<VQZ_MIPivE|C*CkR92O"
    b"6<LrOnUE0~kRIuf7HN<gsgM#WkQ~X76iJX6iI5Np5Fhan7jY09u@Dn65FODF6;Tiwkq{9P5FX(W76uv$9$felf-ndMKg7Ui9"
    b"2BU~_!s{mR8XkUfAAZ>@Do4q9pCU3U+@{9@DU&I9`EoLZ}8fBg_n4N=Xi#vc!I}xgok*5`?!a@xP#lcg`2p6>$rxixME$#C0"
    b"xV>oX0tw#TlH&DV)R!9LF&nwT|F04&fjUU_bU@Z|FT@H+Eqsc3`{MhOJ@?He(YuVuM&8dR^$XVhvVf6;@(}SdL|4DVAU{7Ga"
    b"@S5PE*-"
    b"d15Z+U^Zr9rkH{0Vj8Am3MONcn1~5tJjP)x#$dD<6?$ao5n?!oVJL=Ruo#4aVgUN1ANrz?=#5^YCwibex}mG+g3h87I-"
    b"&#Gqn&6Qx=rZTq7_=A1)8IoXo@DHF&d#E8lb+Yhq|H;YNHlvqK2rBYN9HtpfW0<qNsrKq8!Si3`(PvC@D&yxG09AC?X1@5DK"
    b"Dz$d7!;EAk*Wav`V4f$Yd8vLXvIBa_I83`j51AuZA%wMd1ONFkCV8ImH2NQ^{CC=wt(;vuexgV=~AVj>2jBbtbcD2OZ~SrHL"
    b"Ighx1p6$Tm#o^atuhzNsV1ln$C76OM%=s^7|{z0hFq5l1Y-"
    b"{Kd3;)nQdeZyDr1)uRre8dO5$2;*BZ}3{Y!b`lsbMXvM@kBhf9^s*Qfcv;7?&1z^<CeII8{#^ySyypIT*f7F5f^YC=Wte>!D"
    b"(>{C#@4WE{@@-ID*4Cgo8LB_G6#ei#^tE>=HY%Lu|)3Y{eFA7Mrk9Y`}VJ9oC99SS?mzrL_Xf#WE}vORyM=un-Hxe9RMbF~^"
    b"#XSz;z;i0PPSO~n*38I#0BOu%@I!&or}qs1tUv_@dK7>1!@2nJh&Fi;FYf6)(ptv={2dZDN2f$r!gx}uBdEIOg1=pfpoooFl"
    b"CptWcvTB3z$E}EgKXd)V;k!UCypuVUl>Y|RQEoz~rs3EGOny4zOpt7hWDx!iYFUp~;C?iUvlqe}mptvX|ilT@pEDE8ZC?N79"
    b"pU5lnAh*aRaw3PwF0vu3$RaW$lgKDCAiYQ@(jtvWEm9$+NFkCVnMf*<AhAd!5+Z?!FXAEYU*aIPh$UhohKP=6BC3cYB8y0fh"
    b"zKIQ2q(e{0}UlS;R?S9vBHR8;S+ML3bbQ#Q2tl{h)@Cl{1LzL3qQpV@m+iqU-"
    b"1Q>#V7Gmd=T%gcjB#hBVLPFc!?L{xp*d?iYL}%@kl%r55#@k!(DMl+!nXQP29kBaZOwmSHxxOlDH@?i1Xqc&f<(XEl!D(;)H"
    b"cv91};y5pfuYa8Mi&`^7%7*V-"
    b"d?i(O);*kNrK+r(C}MQp|<Y!n;Bda+KdwbqE$VwG4aR$w`niKSwRSS%J<3&jF4U(6G8tvO=0m?dV48JLb~Vyc)TCW}edL@_~"
    b"(7vscOYm68zMv0MP1cqan7%GN{!D5g#Pz(_LML*Hk>LYrKUZSVyVRaYXL|4&8bhbK)j-"
    b"rETFWR9k+KASom1rqih~}c1XeydmjYT8TP&5$rt$L!as3U5NT2@U_LsS>lL{+Pbs4OapilTy5UX&AMMHx}rDkVya5~8>$W)&"
    b"4hL}5`#6toJ6{34&oEAm*mMJ|z3<Ph1dY$B`3A~K6iRz{IQq!;N#S}To6EmDb;B88P)Boj$R5|P+SBoc}QBEE=c#T9WxY!OS"
    b"uv|=E-h$f<1QAFgwL=q9L2qL@)C&CJ2X`zHCT;aDuL>Lh)WMuu1W634`A8DWe`zieMUxoTls6XPj_$7XdAJ%vA?Jr-"
    b"&7wfb5BtD7{;=T1wycKW6Yw^l@DPD-"
    b"@;+c4AJrR%p@<=?i9*Fzmp13RSShvM3aZ}t7*R5;f>R+yi%hn}vQCtw`#X0M&I3rGrQ{tp`LLC3gF>%y7A`Xi~;-"
    b"ENS?HBw0vRCY}c8gtNr`RF3Tie7|u|;eao2-"
    b"pu!(Y~mb=F$3MywXA#7b+0SpJt~VyU%6EEbEzLb1S_FXsJau9#!Z7PG`mF+)tZrirP4nIa}zlf*<ZL5vsUtg&LuUq*{j)<`k"
    b"pFT=$!Yp56^24j#l&>CR%xB6Lqtv*(7^g>Uoht=KcW_7i?Se>m-"
    b"R!6IY)!u4{wrFFuwpv*&trk{utC`i*YGO6E8d(j|0QIeUR$Z%(Rokj%)wF6@)van)RjZ0s*{Wn!v?`!H%2{QtGFEA;lvUCyV"
    b"HLNESw*cPR$;4<RnRJ6<wri`wencGtz1@4D~FZc%4TJ?vRIj|OjbrKgO%P&XQj2$SgEa4R!S>{mE1~ZCAE@RiLFFPhy+%AE1"
    b"nhCietsLVp%b*7*=#EnibWGVnwzhSrM%WR(LC%71lDAwv^>ruI0BvtT0xv<+B_s$O>)!YyJO68Y-"
    b"0eV}6@o=BN2#zMF66E54Y|=9BqoKA89Boq1zkn^)$gd10QLXXYuMn8)Ukd1xM(`{tgxV{V&U=BBw}uA6J-"
    b"Dz2Ez=90N+E|~M?oH=XGnA7HzIcZLq<K`HSnj_}0Ib;r+17^S3YxbDkW|!G%c9`vE8@8G)X0zF3Hku7)y;*D4nAK*LS!q_7<"
    b"z^X{nk8njS!5QP1!lgPXXctYX11ASW||phI;NSaW{R0?CYgz5f*EhdnXzV!8Er<Hk!A#jn_*_C8Da*TL1v)oZ~B?OrjO}udY"
    b"PW42fCYXrmN{<I-5?Wqv>GUn|7wHX=7TOR;DFdnC7OLX=<97#-@>JXd0OMrk<&5>X_Q57HXOrrn;$Ss+uaMvZ-"
    b"V$nhK`8DQC)>GNv?2nUbc2DQ=3HqNa!`Yzmozrhv(B@|nCQ4|1DaCa1|^vYTustI1+An@lF7$zal(bS5p*nA9efNoi7;<R+O"
    b"(YLb}5CXq>K5}5cV9^#rfCbo%XVwxBxI-;4VCW?t{BAJLL0>YbcCaf_=8)ZD>8ovoKVGwM5#$l}znrnTajSHp!=->LM{-"
    b"M9?Z}_Ue=+F9-{-"
    b"{6b_jsq@>Noneex+aP7kI9p>8JXMeyktqhj^gx>wEf+zO8TRo4BE`>udU|zM?PdOSq^n==1uVKC92@(>SG1>J$37KBkZABRH"
    b"%N>4W-!-mmxRz1X97>s@-M-l4baZP==}=*@bQ-"
    b"l#X|^;oCZ>NR?`UZq#+6<Ds9>7{y!UaS}Cg;=2H>v?*vo}*{$S(vG3=;?Zzo~ozl$(W=k>Ir(h9;e6ZF&M2!>5+Pb9<GP!p%"
    b"|hE>p^;;9-#Z{e(0<F=-"
    b"#@Q?x}m|?&zkw>Mpvo?xZ{F4rs62>9)F!ZmnDCmS~}y>t?#CZlW9OMrf!T==!>zuB+?l+Nh;#>KeMbuBNN%DyXb0>58bJ%j<"
    b"F|tIO!px|A-dOX%V#ri<z#y09*!3+e*Muk-1=I*-"
    b"n+bLpJOp|k63$f~pG%*doO>I^!)PN&oAG)S#e>6AK!POg(7sZOF3>qJPX6X^Ilo{oz+I<}6bV<Lu*uA}Lwh@vCwNIIg9fbcr"
    b"3HabvBd)n20gy=9jn4n5v%^t+~{9m~agi?RhZ}kg5)erSueN$iYMSWJE)JJ?!@6|i?7H`yR^-8@|FYsJFQ%}_sJXVj?L-"
    b"hdn)jf4r-BGu3OWjmA)OB1_SJf4D8JE;WbwQn1=Wte?QK!`@oKz>&adiww)e&`A9a0BzK<!ui)L!gSyVWkW6Fby)wM}hRTd-"
    b"MeQXAC<tXJ#QTD1nN)he}8tx(Ic3`^A#ELMxuLbU+%)jTy<%~7*4OU+a>)O1W!Q`HnT8I#mRH9?J6<1ki@QKQu;j8r4ka5W4"
    b"=)etpU4N?O!0R2@z^i_RSZ`BJuRS(r&byHo@MRiu4R7Z4B?NvL~7Hw2()k?KgEzlgzR8urjja4Jn5Dip)RZrDbbx>Q?QZ-"
    b"c#R9DqdRaH@yQAt%)6;NK4Q)N{dlvbrsQk77}QA`z8MNn83R0ULi<WqT(N99(zkW=MQ*^y0URasPKWKtQCL8VvekXEHpsgX*"
    b"hR4G()BvVO|L?u><kWeL1@exnORdG~o#8NR4Lq%875LHD{kr7EnR1s8ogi~Q*lvaWAl&ky*QDIcDA~HqXBd{tC&7d-HEe+-"
    b"U@qXi%_tX2~eaAQNE53N2y-"
    b")b)eemAno%hyz<Gsc!?<HP%&%I}O>OJuu;}ITu4{+bRhr8Y#?>26EH@zFU?p^b)dRK7SyM&A01@Ao0d1t*dIPIPCPI@PB+&h"
    b"M&-VyIG4tWQ?1K5v!-d^nSc4L>f)7ydV-ZpP5ws@PpP2NUq@YZ9Ux7J&O)!r&^C01a$w+u_YC0Oh&@)lx&H{Y9wx!!Ed@@8U"
    b"&H{F|tsooS!_9kJXHv!|laTx23@kV2mHxeVf;TYx(#Sm{W26+R$0qF1bLtn2CdV4+5!|RT2URSRRI(wba(d&TrUOTk)+Mu=9"
    b"%4>-hUUM|_nxcu<7>&G!XyDcN>Y=Vz2erLgsOi-}b*~z#dR4s2sN_{d1+P5Hd1X<?D~(cKNv{Nod&N-HD}usaAr$lqAitN-"
    b"%Zof-ZshWEB8Qh9*}SaC;$`+SA)}W8>AiGF>!m?zFBMXHDZJ!J<|RcEFEJ8%36a2yk9b~OFAidRu@KXXf#_Z|MD?N|vKPsVh"
    b"zMSIg!96}cpR&q@&e_0elG-JykPhO;U!O^4=g4FqxXMk6#wA2`wKtaANcNm!&mnUKD(dXkNDue$2<2e-"
    b"ng&bS9s~Zz;pK*p5lr77?0eCc!2xvJ=}Hg;5Kf#H*v$g?p|}R;);72m)wiEfb;G-"
    b"oORFOG)}oEal$>0V>s#_!D081d(b_A{q8>Ob@yO5cDXyT!`+T;*y?V<W_J@dVuQOL>)f^O8h15Txht{4U5;f~>Mp@zcM%q1f"
    b"jb}b+_{*8+3qaNbZ1~Xrnyrw#hr{vnCMQxcy}DeVvIW)quh}gf#L2j40VUNgWW+G=ng=Cw;%eVkJ}r)+@9!x?rt}9b-"
    b"SQ5I=LOu!EKLrXzR8?Yqu3zqJ`TW&D^GFg2rwmG;|xFKI*x3QOB)~TBzyPKy|kos-lWp8I|0MsDScrIh1wFpfpOkB~ijHE{e"
    b"HDQ3Qoi$SsHhZhqu*^CFL%8@Z4ZIo#~X=4M3}H#0K18Ib|$k<LwvG;V67a#JFOn;glI6iM8~NaQ9&0yjS5xp5H(u@TFSi5PB"
    b"lL=#cnD2R+mi0DQ@csCrvx-_t^b~)}{J`guhem4YR++g?u@jjwKfyI4bF%=j?hVuWxZ~rg+6hHjm@eN<`#s3+f{2%eb{~quB"
    b"Z}A4N@yh=aFZ|E(Og!~J!DBqaL;nNZ_us=^{~g@+-@;AYz;*vMT=ieUWpT-"
    b"W5f^YC=lo}J#(x^8{3mh3e;mhf6i57ramarV2gH8=KJ3LF?Dp@%PX7*U7u)<>u?3s4$-faB{OhsKzZPq-"
    b"8ms&(u>#An%)b;%uo#Q5(7yolF%NV7b1)mTFcUNU(=iQGF~vU_6EOkfG0s00V=x+{{39^}!!Zm){X;MqgD}uP0R7PqebL9?8"
    b"@<pIJ^bC#4PDU%o&BBA5gpLp-wtij2CdP`-"
    b"x4j*9L@Yq(FBdr2o3!WP#^VB*Ix&<Q42Lu!(ScMP!(1Dl~D;5Q32)s<xm!7P}*M#B~b#!QOsWyMNk-"
    b"p`~^_}`H>HK{dtfZxscPJ!^)0q$m-"
    b"96%*cd{{tQTubV!Rd{?tf?lt|%Ej$}xRBuMN}goH?d`2KhzF5)0IVj(7CAUdM?qxz#DG9n=&A|O1%A*|m7TK+(*<>&Y0=i~D"
    b"SDg<E=j6hJGh;1O87l_WtYB;b42n;g*g!~Tqg`fC=@A!tV_=3;)gpc@u_jrf5c!Sp=uR>np1)k#>p5h4};}IU>0q)}-"
    b"?&1z^<5tMckQ=y;Yq*LlxQt7<hzmH6b2y7LIE_;{8FC`zIF8{cj^Hp3;UErRKlWiS_Fy-5VJCKAJGNmfwqP?hVIww#tPfd-w"
    b"OE7IScR2Xf#q0+rC5T+ScHXGfccn*xtN35n1z{`f$1UBFcniU8Iv#(6EGg*FcxDl8lx~0BQPApFcd>D7=thn1JED+&=-"
    b"Bs8@<pIJ<uK9&=p<K8J*A(9nc=_&=zgb8m-"
    b"V0Ezlgz&=gJ37>&>n4NxETP#1Mj8?{gqHBcSZP!&~B8I@2G6;K}KP!?rS8l_MYB~Tp2P!vT_7==&}1&|;4kQaH78@Z4ZIglO"
    b"MkQG^w8JUm~8IayehqOq8)JTPtNP*-?hNMV>#7KmMNPzf=hq#D?*ocLgh=J&chNy@lA|nzaB0@-"
    b"bghN;u!395ri&dZ%4Uu1M;A0O7R2UHq9;O5$^@MQ)LAgK(B(U=PZ)o`kzwrw{!~DQ^e8X4q1)uQ=AH@f}$2;*BZ}1wg!o0)_"
    b"JjXNf6i@INkHkYfz<qHKcX0=|!`#A6+`x5l4Oejmm&GMq#07C4=WrHh!koq_oWu!n9LI1JN5o+q!a;EW`>_vu!|cIs?7~j51"
    b"KY6;Tg4V^#wM{58?YYh#9FMuYOx9{u>#A(EW=VP!D6up3$Xz6F%NUa9L&Zn%oH;)9n-"
    b"{AOu=MK5)&~2<Hb0P#Tbkhqc9R9#BdD5Pz(`+!wd>D5CcSi^h00t5xvn1Jw*?6M>lj8UC<ewL`QT$d$bd6(FUzWE3`xlG#Aa"
    b"#6iq~9G(tl(5cN?HbwwT2MlIA7HBcSZL{(HlWmFOsQ32&eIg~{iloq8>5+y`&6hl!I5rt6*1w{emM?T~gd5{~qL{8*Dc4QM-"
    b"kp-DWCS*hgq!;PJq(vH$8mUA|q(E{cLsF3hiA5qLL;}P|JP{XhL~O)DOe=<nj%Xq(q98INA)<(Y@FE<-"
    b"A~2R_m>5P0F1}%05oj#}M@^uD;S+%eJQDp6#BwEa76=#w)+2w!Z~PKJ@dMwjZ{jPyh|l<hkNAN1;vL?KH+YR#c!?L{Ii889c"
    b"!I~)Bk>Rq#C_buUEINKaSJ!a4P3`HT*Vb}8JEOGT)=tjoH&a!;xta-Bu?PCIEJI*2oB?rbr1){e(V!_#h&2Z*oB>92eylCVy"
    b"oDK&0-"
    b"TaiVb3Y@H(u;8nGIy#7eP3EXOjj6idWnu_$;U7GS=Zhq+>om@Q^urkH{0Vw#v5JOz_6Nle59F<y)lV=+dI#wamTj1a>yObo>"
    b"kF<1-=9*6<xFZ!Xc=p%ZIUg#-"
    b"$pu6ZMx{5C7EIOg1=pfn$w?kXB5v|cmv=l8wb2Jl8(L^*BjYLB<5cN?{)D?AtYonH^i5j9hs)?$iil~fAq9Q7Y@}it5i!!1#"
    b"N{N!9geZ<;q9}@p!lIBUhyo%%@`=16kI0Q&A}4Z)>>``UiYy{CGKq}A8IWG2Lt2prsYNQ0Qlvm~kqk*i5|LOWLPC)M@kKllS"
    b"HwYV5eqRz3=v&KLsSt3kwqjCQA9v^5e{L65n3?F4wlhhV6YJ^SF2!Ywu0pl3=Wh$1`a%843-"
    b"B_!SX!ezu>I|t^QjR|G{tZ%lavP;Jf%HzT%7cEI#3*_#ocnop>wW;I()qUgCv#E}r43cp@I-"
    b"k$5N`;J&yg?*88$aT~Y9O>qO)#WisiSHxv;2^Yl$aUSQyS#buZ#VK(TC&Y1a3`fNgaTtfhL2=;!_KSVkEB1)p*d=y~9oR0mi"
    b"LKZoHj7QzC^m@oSSQwsHCQcHiIxAiLM+EJu~aO<VzEdp!~!v2%=69rA4S82N&"
)


@lru_cache(maxsize=1)
def amd_rsqrtss_table() -> np.ndarray:
    raw = zlib.decompress(base64.b85decode(_AMD_RSQRTSS_TABLE_B85))
    if len(raw) != 16384:
        raise RuntimeError("invalid AMD RSQRTSS table length")
    if hashlib.sha256(raw).hexdigest().upper() != AMD_RSQRTSS_TABLE_SHA256:
        raise RuntimeError("AMD RSQRTSS table hash mismatch")
    return np.frombuffer(raw, dtype=">u2").astype(np.uint16)


def amd_rsqrtss_initializer() -> str:
    values = amd_rsqrtss_table()
    return ",".join(f"0x{value:03X}u" for value in values)
