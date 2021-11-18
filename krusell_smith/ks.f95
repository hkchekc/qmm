MODULE KSPARAM
    IMPLICIT NONE
    INTEGER:: I, J ! FOR ITER
    REAL(KIND=8), PARAMETER:: BETA=.917493, ALPHA=0.36, DELTA=0.06, GAMMA=2 
    REAL(KIND=8), DIMENSION(2), PARAMETER:: PROD_SHOCK=(/1.0165, 0.9839/)
    REAL(KIND=8), DIMENSION(3), PARAMETER:: EMP_STATE=(/0.715494756670886, 1. , 1.39763428128095/)
    ! DROP: THE NUMBER OF PERIODS TO DROP INIT DEPENDENCE
    INTEGER, PARAMETER:: N=500, T=11000, DROP=1000

    REAL(KIND=8), PARAMETER:: K_MIN=0.01, K_MAX=15.01 ! DEFINE SMALL K GRIDS
    REAL(KIND=8), PARAMETER:: AGG_K_MIN=1.0, AGG_K_MAX=15.0
    INTEGER, PARAMETER:: NZ=SIZE(PROD_SHOCK), NY=SIZE(EMP_STATE), NK=30, NAK=10
    REAL(KIND=8), PARAMETER:: KSTEP=(K_MAX-K_MIN)/FLOAT(NK-1), AGG_KSTEP=(AGG_K_MAX-AGG_K_MIN)/FLOAT(NAK-1)
    REAL(KIND=8), DIMENSION(NK),PARAMETER:: K_GRID =(/(I*KSTEP, I=1,NK)/) + K_MIN - KSTEP
    REAL(KIND=8), DIMENSION(NAK), PARAMETER:: AK_GRID =(/(I*AGG_KSTEP, I=1,NAK)/) + AGG_K_MIN - AGG_KSTEP
    INTEGER, DIMENSION(T):: AGG_SHOCK_VEC
    INTEGER, DIMENSION(N, T):: IND_SHOCK_VEC
    REAL(KIND=8), DIMENSION(6,6):: TRANS_MAT

    REAL(KIND=8), DIMENSION(NAK, NZ):: INTEREST, WAGE
    CONTAINS
        SUBROUTINE CAL_PRICE()
            INTEGER:: ZIDX, AKIDX
            REAL(KIND=8):: NORM_K
            ! UNRELATED TO INDIVIDUAL PROBLEMS
            DO AKIDX=1, NAK
                DO ZIDX=1, NZ
                    NORM_K = AK_GRID(AKIDX) ! /LAB_SUP(ZIDX)
                    INTEREST(AKIDX, ZIDX) = ALPHA*PROD_SHOCK(ZIDX)*(NORM_K**(ALPHA-1.))
                    WAGE(AKIDX, ZIDX) = (1.-ALPHA)*PROD_SHOCK(ZIDX)*(NORM_K**ALPHA)
                ENDDO
            ENDDO
        END SUBROUTINE

        SUBROUTINE INIT_ARRAY()
            OPEN(UNIT=11, FILE='input/markov.txt', STATUS='OLD')
            DO I=1, NY*NZ
                READ(UNIT=11, FMT=*) TRANS_MAT(:, I) ! CHECK DIRECTION
                PRINT*,"trans", TRANS_MAT(:, I)
            ENDDO
            CLOSE(UNIT=11)
        END SUBROUTINE
    END MODULE

MODULE KSRES
    USE KSPARAM
    IMPLICIT NONE
    REAL(KIND=8), DIMENSION(NK, NY, NAK, NZ):: VFUNC=0., VFUNC_TMP
    REAL(KIND=8), DIMENSION(NK, NY, NAK, NZ, NK):: VFUNC_NEW
    INTEGER, DIMENSION(NK, NY, NAK, NZ):: PFUNC, PFUNC_NEW
    REAL(KIND=8), DIMENSION(2):: INTERCEPT=(/0.095, 0.085/), SLOPE=(/0.99, 0.99/)
    REAL(KIND=8):: R_SQ=0., ERROR=100.
    INTEGER, DIMENSION(T-DROP):: AGG_SHOCK_VEC_TRIM
    INTEGER, DIMENSION(N, T-DROP):: IND_TRIM
    REAL(KIND=8), DIMENSION(NAK, NZ):: KPR_EST
    REAL(KIND=8), DIMENSION(NK, NY, NAK, NZ, NK):: CONSUM, UTIL
    REAL(KIND=8), PARAMETER:: CRIT=1e-1
    REAL(KIND=8), DIMENSION(T):: SIM_AK
    REAL(KIND=8), DIMENSION(N,T):: SIM_SMALLK
    REAL(KIND=8), DIMENSION(NZ):: R_SQUARE
END MODULE

PROGRAM KS
    USE KSPARAM
    USE KSRES
    IMPLICIT NONE
    REAL:: START, FINISH  ! FOR TIME
    CALL CPU_TIME(START)
    CALL INIT_ARRAY()
    CALL CAL_PRICE()
    CALL INIT_SHOCKS()
    DO WHILE (ERROR>CRIT .OR. MAXVAL(R_SQUARE)<0.8)
        CALL VFI()
        CALL PSEUDO_PANEL()
        CALL CAL_ERRORS()
    ENDDO
    CALL CPU_TIME(FINISH)
    PRINT '("Time = ",f6.3," seconds.")',FINISH-START
    CALL WRITE_ALL()

END PROGRAM

SUBROUTINE VFI()
        USE KSPARAM
        USE KSRES
        USE OMP_LIB
        IMPLICIT NONE
        INTEGER:: KIDX, AKIDX, ZIDX, YIDX, KPIDX
        REAL(KIND=8):: ERROR_VFI

        ! FOR PARALLEL CALCULATION, REDUCE PRIVATE VARIABLES IN MAIN LOOP
        ! PRE CALCULATE THE KPR, CONSUM, UTILTIY
        ! ESTIMATION OF
        DO AKIDX=1,NAK
        DO ZIDX=1, NZ
            KPR_EST(AKIDX, ZIDX) = EXP(INTERCEPT(ZIDX)+SLOPE(ZIDX)*LOG(AK_GRID(AKIDX)))
            IF (KPR_EST(AKIDX, ZIDX)<AGG_K_MIN) THEN
                KPR_EST(AKIDX, ZIDX) = AK_GRID(1)
            ELSE IF (KPR_EST(AKIDX, ZIDX)>AK_GRID(NAK)) THEN
                KPR_EST(AKIDX, ZIDX) = AGG_K_MAX
            ENDIF
        ENDDO
        ENDDO

        UTIL = -1e12
        !$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(AKIDX, ZIDX, KIDX, YIDX)
        DO AKIDX=1,NAK
        DO ZIDX=1, NZ
        DO KIDX=1,NK
        DO YIDX=1,NY
        DO KPIDX=1,NK ! NEXT PERIOD K CHOICE
            CONSUM(KIDX, YIDX, AKIDX, ZIDX, KPIDX) = (1+INTEREST(AKIDX, ZIDX)-DELTA)*K_GRID(KIDX)&
            +WAGE(AKIDX, ZIDX)*EMP_STATE(YIDX)-K_GRID(KPIDX)
            IF (CONSUM(KIDX, YIDX, AKIDX, ZIDX, KPIDX)>0.) THEN
                UTIL(KIDX, YIDX, AKIDX, ZIDX, KPIDX)= (CONSUM(KIDX, YIDX, AKIDX, ZIDX, KPIDX)**(1-GAMMA) -1)/(1-GAMMA)
            ELSE
                EXIT   ! IF THIS KP IS LARGER THAN INCOME, ALL NEXT ONE EXCEED
            ENDIF
        ENDDO
        ENDDO
        ENDDO
        ENDDO
        ENDDO
        !$OMP END PARALLEL DO

        ERROR_VFI=100.
        VFUNC_NEW = -1e12
        VFUNC=UTIL(:,:,:,:,1)
        DO WHILE (ERROR_VFI> CRIT) ! VFI
            CALL BELLMAN()
            ERROR_VFI = MAXVAL(ABS(VFUNC_TMP-VFUNC))
            PRINT*, ERROR_VFI, MAXLOC(ABS(VFUNC_TMP-VFUNC)), "HIHI"
            VFUNC = VFUNC_TMP
        ENDDO
END SUBROUTINE

SUBROUTINE BELLMAN()
    USE KSPARAM
    USE KSRES
    USE OMP_LIB
    IMPLICIT NONE
    INTEGER:: AKIDX, ZIDX, KIDX,YIDX, ZZI,YYI, ROWIDX, NEXTROW, KPIDX
    REAL(KIND=8), DIMENSION(1):: KPR, SMALL_KPR ! , NEXTUGE, NEXTUGN, NEXTUBE, NEXTUBN
    REAL(KIND=8), DIMENSION(1):: NEXTU
    REAL(KIND=8):: EXPU

    ! DEFINE SOME INTERPOLATIONS
    ! $OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(AKIDX, ZIDX, KIDX,YIDX, ZZI,YYI, ROWIDX, NEXTROW, KPIDX, KPR, SMALL_KPR, NEXTU,EXPU)
    DO AKIDX=1,NAK
    DO ZIDX=1, NZ
    DO YIDX=1,NY
    DO KIDX=1,NK
        KPR(1) = KPR_EST(AKIDX, ZIDX) ! off grid
        DO KPIDX=1, NK
            SMALL_KPR(1) = K_GRID(KPIDX) ! this is on grid
            EXPU = 0.
            ROWIDX =ZIDX + 2*(YIDX-1)
            DO ZZI=1, NZ
            DO YYI=1,NY ! SHOULD USE 4 INTER2D INSTEAD OF INCREMENT LOOP FOR PARALLEL
                NEXTROW = ZZI + 2*(YYI-1)
                CALL PWL_INTERP_2D(NK, NAK, K_GRID, AK_GRID, VFUNC(:,YYI, :, ZZI), 1, SMALL_KPR, KPR, NEXTU) !  1d is enough
                EXPU = EXPU+TRANS_MAT(ROWIDX,NEXTROW)*NEXTU(1)  ! += increments on expu
            ENDDO
            ENDDO
            VFUNC_NEW(KIDX, YIDX, AKIDX, ZIDX,KPIDX) = UTIL(KIDX, YIDX, AKIDX, ZIDX, KPIDX)+ BETA*EXPU
        ENDDO
        PFUNC(KIDX, YIDX, AKIDX, ZIDX) = MAXLOC(VFUNC_NEW(KIDX, YIDX, AKIDX, ZIDX, :),1)
        VFUNC_TMP(KIDX, YIDX, AKIDX, ZIDX) = MAXVAL(VFUNC_NEW(KIDX, YIDX, AKIDX, ZIDX, :),1)
    ENDDO
    ENDDO
    ENDDO
    ENDDO
    ! $OMP END PARALLEL DO

END SUBROUTINE

SUBROUTINE INIT_SHOCKS()
    USE KSPARAM
    USE KSRES
    IMPLICIT NONE
    INTEGER:: TIDX, NIDX, Z_PREV
    REAL(KIND=8):: RAND_TMP, THIS_ROW
    REAL(KIND=8):: TRANS_PROB
    INTEGER, PARAMETER:: RSEED=4200
    INTEGER:: OTHER_STATE, CUR_SHOCK, LAST_ROW, NEXT_STATE, LAST_ASHOCK

    CALL SRAND(RSEED)

    ! FIRST ONE, DETERMINISTIC EVERY RUN (WILL DROP ANYWAY, IS OK)
    AGG_SHOCK_VEC(1) = 1 ! STORE INDEX INSTEAD OF VALUE OF SHOCK HERE, START GOOD
    ! AGGREGATE SHOCKS
    DO TIDX=2,T
        Z_PREV = AGG_SHOCK_VEC(TIDX-1)
        CALL RANDOM_NUMBER(RAND_TMP)
        NEXT_STATE = Z_PREV
        IF (RAND_TMP<1./8.) THEN ! RANDOM NEXT STATE (SYMMETRIC)
            OTHER_STATE =1
            IF (Z_PREV==1) THEN
                OTHER_STATE = 2
            ENDIF
            NEXT_STATE = OTHER_STATE
        ENDIF
        AGG_SHOCK_VEC(TIDX) = NEXT_STATE
    ENDDO


    IND_SHOCK_VEC(:,1) = 1 ! EVERYONE EMPLOYED STATE
    ! INDIVIDUAL SHOCKS
    ! WRONG
    DO TIDX=2,T
        CUR_SHOCK = AGG_SHOCK_VEC(TIDX)
        LAST_ASHOCK = AGG_SHOCK_VEC(TIDX-1)
        DO NIDX=1,N
            CALL RANDOM_NUMBER(RAND_TMP)
            TRANS_PROB = RAND_TMP
            LAST_ROW = LAST_ASHOCK + 2*(IND_SHOCK_VEC(NIDX, TIDX-1)-1)
            IF (CUR_SHOCK==1) THEN
                THIS_ROW = TRANS_MAT(LAST_ROW,1)/(TRANS_MAT(LAST_ROW,1)+TRANS_MAT(LAST_ROW,3))
            ELSE
                THIS_ROW = TRANS_MAT(LAST_ROW,2)/(TRANS_MAT(LAST_ROW,2)+TRANS_MAT(LAST_ROW,4))
            ENDIF
            IF (THIS_ROW>TRANS_PROB) THEN ! IF MORE THAN, THEN UMEMPLOY
                IND_SHOCK_VEC(NIDX, TIDX) = 1
            ELSE
                IND_SHOCK_VEC(NIDX, TIDX) = 2
            ENDIF
        ENDDO
    ENDDO

    ! NEED TRIMMED TO DROP INITIAL DEPENDENCE
    AGG_SHOCK_VEC_TRIM = AGG_SHOCK_VEC((DROP+1):T)
    IND_TRIM = IND_SHOCK_VEC(:,DROP+1:T)
END SUBROUTINE

SUBROUTINE PSEUDO_PANEL()
    USE KSPARAM
    USE KSRES
    USE OMP_LIB
    IMPLICIT NONE
    REAL(KIND=8), PARAMETER:: INIT_K=8
    INTEGER:: TIDX, ZIDX, YIDX, NIDX, AKIDX
    REAL(KIND=8), DIMENSION(NK, NY, NAK, NZ):: KDECISION
    REAL(KIND=8), DIMENSION(1):: KPR, SMALL_KPR, SMALLK_TMP

    ! DON'T NEED TO WORK WITH SHOCKS AGAIN, ALWAYS WORK WITH SAME SET OF SHOCKS
    SIM_AK = 0.
    ! FIND AGGREGATE CAPITAL, USING THE THE FUNCTIONAL FORM
    SIM_AK(1) = INIT_K


    ! POPULATE SMALL K, USING THE POLICY FUNCTION FOUND IN BELLMAN
    DO YIDX=1,NY
    DO ZIDX=1,NZ
    DO AKIDX=1,NAK
    KDECISION(:,YIDX, AKIDX, ZIDX) = K_GRID(PFUNC(:,YIDX, AKIDX, ZIDX))
    ENDDO
    ENDDO
    ENDDO
    SIM_SMALLK(:,1)=INIT_K
    DO TIDX=2,T
    DO NIDX=1, N
        ZIDX= AGG_SHOCK_VEC(TIDX)
        YIDX= IND_SHOCK_VEC(NIDX,TIDX)
        KPR(1)= SIM_AK(TIDX-1)
        SMALL_KPR(1) = SIM_SMALLK(NIDX, TIDX-1)

        CALL PWL_INTERP_2D(NK, NAK, K_GRID, AK_GRID, KDECISION(:,YIDX,:,ZIDX), 1, SMALL_KPR, KPR, SMALLK_TMP)
        SIM_SMALLK(NIDX, TIDX) = SMALLK_TMP(1)
    ENDDO
        SIM_AK(TIDX) = SUM(SIM_SMALLK(:,TIDX),1)/N
    ENDDO
END SUBROUTINE


SUBROUTINE CAL_ERRORS()
    USE KSPARAM
    USE KSRES
    IMPLICIT NONE
    REAL(KIND=8), DIMENSION((NZ*2)):: ERROR_PARA
    REAL(KIND=8), DIMENSION(2)::INTERCEPT_NEW, SLOPE_NEW
    REAL(KIND=8), DIMENSION(:,:), ALLOCATABLE :: SIM_GOOD, SIM_BAD
    INTEGER:: NGOOD, NBAD, COUNT_GOOD, COUNT_BAD
    INTEGER:: TIDX, IDX
    REAL(KIND=8):: R1, R2, R3, MEAN, S_RES

    ! CALCUALTE NEW INTERCEPT AND SLOPE FOR FUNCTIONAL FORMS
    ! 1. SEPARTE INTO GOOD AND BAD STATE
    NBAD = SUM(AGG_SHOCK_VEC_TRIM(:T-1-DROP)-1)
    NGOOD = SIZE(AGG_SHOCK_VEC_TRIM(:T-1-DROP)) - NBAD
    ALLOCATE(SIM_GOOD(NGOOD,2)) ! 1 IS X, 2 IS Y
    ALLOCATE(SIM_BAD(NBAD,2))
    SIM_BAD=1.
    SIM_GOOD=1.
    COUNT_GOOD=1
    COUNT_BAD=1
    DO TIDX=DROP+1,T-1
        IF (AGG_SHOCK_VEC(TIDX)==2) THEN ! IF BAD
            SIM_BAD(COUNT_BAD, 1) = SIM_AK(TIDX)
            SIM_BAD(COUNT_BAD,2) = SIM_AK(TIDX+1)
            COUNT_BAD = COUNT_BAD +1
        ELSE
            SIM_GOOD(COUNT_GOOD,1) = SIM_AK(TIDX)
            SIM_GOOD(COUNT_GOOD,2) = SIM_AK(TIDX+1)
            COUNT_GOOD = COUNT_GOOD +1
        ENDIF
    ENDDO

    ! 2. RUN OLS AND GET R SQUARE AND THE INTERCEPTS -GOOD
    MEAN = SUM(LOG(SIM_GOOD(:,2)))/COUNT_GOOD
    R1 =SUM((LOG(SIM_GOOD(:,2))-MEAN)**2)
    R2 =SUM((LOG(SIM_GOOD(:,2))-MEAN)*(LOG(SIM_GOOD(:,1))-SUM(LOG(SIM_GOOD(:,1)))/N))
    R3 =SUM((LOG(SIM_GOOD(:,1))-SUM(LOG(SIM_GOOD(:,1)))/N)**2)
    PRINT*, "RS", R1, R2, R3, MEAN
    SLOPE_NEW(1) = R2/R1 ! BETA = (X'Y)/(X'X)
    INTERCEPT_NEW(1) = (SUM((LOG(SIM_GOOD(:,2)))) - SLOPE_NEW(1)*SUM((LOG(SIM_GOOD(:,1)))))/NGOOD
    S_RES = 0.
    DO IDX=1, NGOOD
        S_RES = S_RES+ SIM_GOOD(IDX,2) - INTERCEPT_NEW(1)-SLOPE_NEW(1)*SIM_GOOD(IDX,1)
    ENDDO
    R_SQUARE(1) = 1- S_RES/R3 ! R_SQUARE
    ! SAME THING FOR BAD STATES
    MEAN = SUM(LOG(SIM_BAD(:,2)))/COUNT_BAD
    PRINT*,"MEANB", MEAN, SUM(LOG(SIM_BAD(:,2)))
    R1 =SUM((LOG(SIM_BAD(:,2))-MEAN)**2)
    R2 =SUM((LOG(SIM_BAD(:,2))-MEAN)*(LOG(SIM_BAD(:,1))-SUM(LOG(SIM_BAD(:,1)))/N))
    R3 =SUM((LOG(SIM_BAD(:,1))-SUM(LOG(SIM_BAD(:,1)/N)))**2)
    SLOPE_NEW(2) = R2/R1
    INTERCEPT_NEW(2) = (SUM((LOG(SIM_BAD(:,2)))) - SLOPE_NEW(1)*SUM((LOG(SIM_BAD(:,1)))))/NBAD
    S_RES = 0.
    DO IDX=1, NBAD
        S_RES = S_RES+ SIM_BAD(IDX,2) - INTERCEPT_NEW(1)-SLOPE_NEW(1)*SIM_BAD(IDX,1)
    ENDDO
    R_SQUARE(2) = 1- S_RES/R3 ! R_SQUARE

    ! CALCULATE ERROR
    ERROR_PARA(1:2) = SLOPE_NEW-SLOPE
    ERROR_PARA(3:4) = INTERCEPT_NEW-INTERCEPT
    PRINT*,"RSQ", R_SQUARE
    PRINT*,"ERROR_PARA", ERROR_PARA
    PRINT*, "PARAS", INTERCEPT_NEW, SLOPE_NEW
    PRINT*, "=============================================================="
    ERROR = MAXVAL(ABS(ERROR_PARA))
    INTERCEPT=INTERCEPT_NEW
    SLOPE=SLOPE_NEW
END SUBROUTINE



subroutine pwl_interp_2d ( nxd, nyd, xd, yd, zd, ni, xi, yi, zi )

!*****************************************************************************80
!
!! PWL_INTERP_2D: piecewise linear interpolant to data defined on a 2D grid.
!
!  Discussion:
!
!    Thanks to Adam Hirst for pointing out an error in the formula that
!    chooses the interpolation triangle, 04 February 2018.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    04 February 2018
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) NXD, NYD, the number of X and Y data values.
!
!    Input, real ( kind = 8 ) XD(NXD), YD(NYD), the sorted X and Y data.
!
!    Input, real ( kind = 8 ) ZD(NXD,NYD), the Z data.
!
!    Input, integer ( kind = 4 ) NI, the number of interpolation points.
!
!    Input, real ( kind = 8 ) XI(NI), YI(NI), the coordinates of the
!    interpolation points.
!
!    Output, real ( kind = 8 ) ZI(NI), the value of the interpolant.
!
  implicit none

  integer ( kind = 4 ):: ni
  integer ( kind = 4 ):: nxd
  integer ( kind = 4 ):: nyd

  real ( kind = 8 ):: alpha1
  real ( kind = 8 ):: beta1
  real ( kind = 8 ):: det
  real ( kind = 8 ):: dxa
  real ( kind = 8 ):: dxb
  real ( kind = 8 ):: dxi
  real ( kind = 8 ):: dya
  real ( kind = 8 ):: dyb
  real ( kind = 8 ):: dyi
  real ( kind = 8 ):: gamma
  integer ( kind = 4 ):: i1
  integer ( kind = 4 ):: j
  integer ( kind = 4 ):: k
  real ( kind = 8 ):: r8_huge= 1.79769313486231571D+308
  integer ( kind = 4 ):: r8vec_bracket5
  real ( kind = 8 ):: xd(nxd)
  real ( kind = 8 ):: xi(ni)
  real ( kind = 8 ):: yd(nyd)
  real ( kind = 8 ):: yi(ni)
  real ( kind = 8 ):: zd(nxd,nyd)
  real ( kind = 8 ):: zi(ni)

  do k = 1, ni
!
!  For interpolation point (xi(k),yi(k)), find data intervals I and J so that:
!
!    xd(i) <= xi(k) <= xd(i+1),
!    yd(j) <= yi(k) <= yd(j+1).
!
!  But if the interpolation point is not within a data interval,
!  assign the dummy interpolant value zi(k) = infinity.
!
    i1 = r8vec_bracket5 ( nxd, xd, xi(k) )
    if ( i1 == -1 ) then
      zi(k) = r8_huge
      cycle
    end if

    j = r8vec_bracket5 ( nyd, yd, yi(k) )
    if ( j == -1 ) then
      zi(k) = r8_huge
      cycle
    end if
!
!  The rectangular cell is arbitrarily split into two triangles.
!  The linear interpolation formula depends on which triangle
!  contains the data point.
!
!    (I,J+1)--(I+1,J+1)
!      |\       |
!      | \      |
!      |  \     |
!      |   \    |
!      |    \   |
!      |     \  |
!    (I,J)---(I+1,J)
!
    if ( yi(k) < yd(j+1) &
      + ( yd(j) - yd(j+1) ) * ( xi(k) - xd(i1) ) / ( xd(i1+1) - xd(i1) ) ) then

      dxa = xd(i1+1) - xd(i1)
      dya = yd(j)   - yd(j)

      dxb = xd(i1)   - xd(i1)
      dyb = yd(j+1) - yd(j)

      dxi = xi(k)   - xd(i1)
      dyi = yi(k)   - yd(j)

      det = dxa * dyb - dya * dxb

      alpha1 = ( dxi * dyb - dyi * dxb ) / det
      beta1 =  ( dxa * dyi - dya * dxi ) / det
      gamma = 1.0D+00 - alpha1 - beta1

      zi(k) = alpha1 * zd(i1+1,j) + beta1 * zd(i1,j+1) + gamma * zd(i1,j)

    else

      dxa = xd(i1)   - xd(i1+1)
      dya = yd(j+1) - yd(j+1)

      dxb = xd(i1+1) - xd(i1+1)
      dyb = yd(j)   - yd(j+1)

      dxi = xi(k)   - xd(i1+1)
      dyi = yi(k)   - yd(j+1)

      det = dxa * dyb - dya * dxb

      alpha1 = ( dxi * dyb - dyi * dxb ) / det
      beta1 =  ( dxa * dyi - dya * dxi ) / det
      gamma = 1.0D+00 - alpha1 - beta1

      zi(k) = alpha1 * zd(i1,j+1) + beta1 * zd(i1+1,j) + gamma * zd(i1+1,j+1)

    end if
  end do
end subroutine

function r8vec_bracket5 ( nd, xd, xi )

!*****************************************************************************80
!
!! R8VEC_BRACKET5 brackets data between successive entries of a sorted R8VEC.
!
!  Discussion:
!
!    We assume XD is sorted.
!
!    If XI is contained in the interval [XD(1),XD(N)], then the returned
!    value B indicates that XI is contained in [ XD(B), XD(B+1) ].
!
!    If XI is not contained in the interval [XD(1),XD(N)], then B = -1.
!
!    This code implements a version of binary search which is perhaps more
!    understandable than the usual ones.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    14 October 2012
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) ND, the number of data values.
!
!    Input, real ( kind = 8 ) XD(N), the sorted data.
!
!    Input, real ( kind = 8 ) XD, the query value.
!
!    Output, integer ( kind = 4 ) R8VEC_BRACKET5, the bracket information.
!
  implicit none

  integer ( kind = 4 ) nd

  integer ( kind = 4 ) b
  integer ( kind = 4 ) l
  integer ( kind = 4 ) m
  integer ( kind = 4 ) r
  integer ( kind = 4 ) r8vec_bracket5
  real ( kind = 8 ) xd(nd)
  real ( kind = 8 ) xi

  if ( xi < xd(1) .or. xd(nd) < xi ) then

    b = -1

  else

    l = 1
    r = nd

    do while ( l + 1 < r )
      m = ( l + r ) / 2
      if ( xi < xd(m) ) then
        r = m
      else
        l = m
      end if
    end do

    b = l

  end if

  r8vec_bracket5 = b

  return
end

SUBROUTINE WRITE_ALL()
    USE KSRES
    USE KSPARAM
    IMPLICIT NONE
    INTEGER:: SROWIDX
    CHARACTER(LEN=130):: PATH="./data_output/"
    CHARACTER(LEN=150):: FILE_NAME

        FILE_NAME = TRIM(PATH)//"VFUNC"
        OPEN(UNIT=1, FILE=FILE_NAME, STATUS='REPLACE') ! START WITH THE TWO VALUE FUNCTIONS
        DO SROWIDX=1, NK
            WRITE(UNIT=1,FMT=*) VFUNC(SROWIDX,:, 8,1)
        ENDDO
        CLOSE(UNIT=1)

        FILE_NAME = TRIM(PATH)//"PFUNC"
        OPEN(UNIT=2, FILE=FILE_NAME, STATUS='REPLACE') ! ALSO SAVE POLICY FUNCTIONS
        DO SROWIDX=1, NK
            WRITE(UNIT=2,FMT=*) K_GRID(PFUNC(SROWIDX,1, 2,1)), K_GRID(PFUNC(SROWIDX,1, 9,1))
        ENDDO
        CLOSE(UNIT=2)

        FILE_NAME = TRIM(PATH)//"SIM_AK"
        OPEN(UNIT=3, FILE=FILE_NAME, STATUS='REPLACE') ! ALSO SAVE POLICY FUNCTIONS
        DO SROWIDX=1, T
            WRITE(UNIT=3,FMT=*) SIM_AK(SROWIDX)
        ENDDO
        CLOSE(UNIT=3)

        FILE_NAME = TRIM(PATH)//"SIM_SMALLK"
        OPEN(UNIT=4, FILE=FILE_NAME, STATUS='REPLACE') ! FOR HAVING THE X-AXIS OF PLOT
        DO SROWIDX=1, T
            WRITE(UNIT=4,FMT=*) SIM_SMALLK(10, SROWIDX)
        ENDDO
        CLOSE(UNIT=4)

        FILE_NAME = TRIM(PATH)//"AGG_SHOCK_VEC"
        OPEN(UNIT=5, FILE=FILE_NAME, STATUS='REPLACE') ! START WITH THE TWO VALUE FUNCTIONS
        DO SROWIDX=1, T
            WRITE(UNIT=5,FMT=*) AGG_SHOCK_VEC(SROWIDX)
        ENDDO
        CLOSE(UNIT=5)

        FILE_NAME = TRIM(PATH)//"INTERESTANDWAGE"
        OPEN(UNIT=6, FILE=FILE_NAME, STATUS='REPLACE') ! START WITH THE TWO VALUE FUNCTIONS
        DO SROWIDX=1, NAK
            WRITE(UNIT=6,FMT=*) INTEREST(SROWIDX,:), WAGE(SROWIDX,:)
        ENDDO
        CLOSE(UNIT=6)

        FILE_NAME = TRIM(PATH)//"CONSUM"
        OPEN(UNIT=65, FILE=FILE_NAME, STATUS='REPLACE') ! START WITH THE TWO VALUE FUNCTIONS
        DO SROWIDX=1, NK
            WRITE(UNIT=65,FMT=*) CONSUM(80, 1, 5, 1, SROWIDX), CONSUM(90, 1, 5, 1, SROWIDX)
        ENDDO
        CLOSE(UNIT=65)

       FILE_NAME = TRIM(PATH)//"UTIL"
        OPEN(UNIT=66, FILE=FILE_NAME, STATUS='REPLACE') ! START WITH THE TWO VALUE FUNCTIONS
        DO SROWIDX=1, NK
            WRITE(UNIT=66,FMT=*) UTIL(80, 1, 5, 1, SROWIDX), UTIL(90, 1, 5, 1, SROWIDX)
        ENDDO
        CLOSE(UNIT=66)

END SUBROUTINE
