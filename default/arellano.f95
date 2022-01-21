MODULE ARRELANO_PARA
    IMPLICIT NONE
    ! ALPHA - EV PARAM, BETA - DISCOUNT FACTOR, TAU - DEFAULT COST
    REAL(KIND=8):: ALPHA=1., BETA=0.8, INTEREST=0.1,  EULERGAMMA = 0.5772, TAU = 0.
    REAL(KIND=8), DIMENSION(2):: STATES = (/.2, 1.2/)
    REAL(KIND=8), DIMENSION(2,2):: EMP_MARKOV = TRANSPOSE(RESHAPE((/0.2,0.8,0.2,0.8/),(/2,2/)))
    INTEGER, PARAMETER:: NZ=SIZE(STATES), NA=150
    REAL(KIND=8),PARAMETER:: A_MAX=2., A_MIN=0., STEP=(A_MAX-A_MIN)/FLOAT(NA-1) 
    INTEGER:: I, MAXIT= 1000 ! ITER
    REAL(KIND=8), DIMENSION(NA), PARAMETER:: A_GRID= (/(I*STEP, I=1,NA)/) + A_MIN - STEP
    LOGICAL:: EGM=.FALSE., EV_SHOCK = .TRUE. ! BOOLEAN FOR POOLING OR SEPARTING EQUILIBRIUM
END MODULE

MODULE ARRELANO_RES     ! FOR RESULTS
    USE ARRELANO_PARA
    IMPLICIT NONE
    REAL(KIND=8), DIMENSION(NA,NZ):: VFUNC_CLEAN=0., VFUNC_CLEAN_NEW=-1.
    REAL(KIND=8), DIMENSION(NZ):: VFUNC_DEF=-10., VFUNC_DEF_NEW=-1.
    REAL(KIND=8), DIMENSION(NA, NZ):: VFUNC_O=-10.
    INTEGER, DIMENSION(NA,NZ):: DFUNC
    REAL(KIND=8), DIMENSION(NA,NZ):: Q= 0.9 ! LIST OF BOND PRICE GIVEN RISK OF DEFAULTING

END MODULE

PROGRAM ARRELANO ! MAIN PROGRAM
    USE ARRELANO_PARA
    USE ARRELANO_RES
    IMPLICIT NONE
    ! FOR LOOPING
    REAL(KIND=8):: ERROR_VFI, ERROR_CLEAN, ERROR_DEF, ERROR_Q=100. ! INITIAL ERRORS TO BE UPDATED
    REAL(KIND=8):: CRIT_VFI=1e-3, CRIT_Q=1e-3 ! CRITICAL TOLERANCE VALUES
    INTEGER:: IT! SOME INDEXES FOR VARIOUS LOOPS

    PRINT*, NA

    ! MAIN LOOP
    IT = 0 
    PRINT*, EMP_MARKOV, "MARKOV"
    DO WHILE (ERROR_Q>CRIT_Q)
        PRINT*, ERROR_Q, "ERROR_Q"
        PRINT*, "====================================================="
        ERROR_VFI = 100.
        DFUNC = 0 ! THIS ARRAY JUST STORE THE CURRENT DECISION
        DO WHILE (ERROR_VFI>CRIT_VFI)  ! START VFI
            ! VFUNC_CLEAN_NEW = -20
            ! VFUNC_DEF_NEW = -20
            CALL BELLMAN_CLEAN() ! START BELLMAN CLEAN
            CALL BELLMAN_DEFAULT(ERROR_CLEAN, ERROR_DEF) ! START BELLMAN DEFAULTED
            ERROR_VFI = MAX(ERROR_CLEAN, ERROR_DEF)
        ENDDO ! END VFI

        ! COMPUTE ERROR AND Q AND NEW Q
        CALL Q_SEPARATING(ERROR_Q)
        IT = IT +1
        IF (IT > MAXIT) THEN
            EXIT
        ENDIF
    ENDDO
    CALL CAL_MOMENTS()
    CALL WRITE_ALL()! WRITE RESULTS FOR PLOTTING USE
END PROGRAM ARRELANO

! ALL SUBROUTINES
! RULE: ONLY ONE OUTER LOOP PER SUBROUTINE
SUBROUTINE BELLMAN_CLEAN()
    USE ARRELANO_PARA
    USE ARRELANO_RES
    IMPLICIT NONE
    REAL(KIND=8):: COND_MAX_UTIL, CONSUM, UTIL, NU
    INTEGER:: SROWIDX, SCOLIDX, CHOICEIDX, NZI
    DO SCOLIDX = 1, NZ ! START BELLMAN CLEAN
        DO SROWIDX=1 , NA
            COND_MAX_UTIL = -1e2
            DO CHOICEIDX=1,NA ! LOOP OVER CHOICE OF ASSET PRIME
                CONSUM =  STATES(SCOLIDX) + Q(CHOICEIDX, SCOLIDX)- A_GRID(SROWIDX)
                IF (CONSUM > 0.) THEN
                    NU = 0.
                    IF (EV_SHOCK) THEN
                        DO NZI=1, NZ
                            NU = NU + EMP_MARKOV(SCOLIDX,NZI)*VFUNC_O(CHOICEIDX, NZI)
                        ENDDO
                    ELSE
                        DO NZI=1, NZ
                            NU  = NU + EMP_MARKOV(SCOLIDX,NZI)*MAX(VFUNC_CLEAN(CHOICEIDX,NZI), VFUNC_DEF(NZI))
                        ENDDO
                    ENDIF
                    NU = BETA*NU
                    
                    UTIL = NU-1./CONSUM
                    IF (UTIL>COND_MAX_UTIL) THEN
                        COND_MAX_UTIL = UTIL
                        VFUNC_CLEAN_NEW(SROWIDX, SCOLIDX) = UTIL
                    ENDIF
                ENDIF
            ENDDO ! END LOOP CHOICE SPACE FOR ONE STATE
        ENDDO
    ENDDO ! END BELLMAN CLEAN
END SUBROUTINE

SUBROUTINE BELLMAN_DEFAULT(ERROR_CLEAN, ERROR_DEF)
    USE ARRELANO_PARA
    USE ARRELANO_RES
    IMPLICIT NONE
    REAL(KIND=8), INTENT(OUT):: ERROR_CLEAN, ERROR_DEF
    INTEGER:: SROWIDX, SCOLIDX
    ! FINANCIAL AUTARKY 
    DO SCOLIDX=1, NZ
        VFUNC_DEF_NEW(SCOLIDX) = -1./(STATES(SCOLIDX)-TAU)+ BETA*SUM(EMP_MARKOV(SCOLIDX,:)*VFUNC_DEF)
    ENDDO
    ! UPDATE DEFAULT DECISIONS
    IF (EV_SHOCK) THEN
        DO SCOLIDX=1, NZ
            DO SROWIDX = 1, NA
                VFUNC_O(SROWIDX, SCOLIDX) = VFUNC_CLEAN_NEW(SROWIDX,SCOLIDX)+&
                EULERGAMMA/ALPHA+LOG(1.+EXP(ALPHA*(VFUNC_DEF_NEW(SCOLIDX) - VFUNC_CLEAN_NEW(SROWIDX,SCOLIDX))))/ALPHA
            ENDDO
        ENDDO
    ELSE
        DO SCOLIDX=1, NZ
            DO SROWIDX = 1, NA
                IF (VFUNC_CLEAN_NEW(SROWIDX, SCOLIDX) > VFUNC_DEF_NEW(SCOLIDX)) THEN  ! CHECK CORRECTNESS
                    DFUNC(SROWIDX, SCOLIDX) = 0
                    VFUNC_O = VFUNC_CLEAN_NEW(SROWIDX, SCOLIDX)
                ELSE  
                    DFUNC(SROWIDX, SCOLIDX) = 1
                    VFUNC_O(SROWIDX, SCOLIDX) = VFUNC_DEF_NEW(SCOLIDX)
                ENDIF
            ENDDO
        ENDDO
    ENDIF
    ERROR_DEF = MAXVAL(ABS(VFUNC_DEF_NEW - VFUNC_DEF))
    VFUNC_DEF = VFUNC_DEF_NEW
    ! NEED TO ADJUST ERROR OF VFUNC CLEAN AS WELL
    ERROR_CLEAN = MAXVAL(ABS(VFUNC_CLEAN_NEW - VFUNC_CLEAN))
    VFUNC_CLEAN = VFUNC_CLEAN_NEW
END SUBROUTINE

SUBROUTINE Q_SEPARATING(ERROR_Q)
    USE ARRELANO_PARA
    USE ARRELANO_RES
    IMPLICIT NONE
    REAL(KIND=8), INTENT(INOUT):: ERROR_Q
    REAL(KIND=8):: CRIT_Q = 1e-3, RISK, QZERO, DIFF
    INTEGER:: SCOLIDX, NEXT_STATE_IDX, CHOICE
    REAL(KIND=8), DIMENSION(NA,NZ):: ERROR_ARR

    ! FIRST CALCUALTE ERROR
    DO CHOICE=1,NA  ! b'
        DO SCOLIDX=1, NZ  ! y (current)
            RISK = 0.0
            IF (EV_SHOCK) THEN
                DO NEXT_STATE_IDX=1, NZ
RISK = RISK + EMP_MARKOV(SCOLIDX, NEXT_STATE_IDX)*1./(1.+EXP(ALPHA*(VFUNC_DEF(NEXT_STATE_IDX)-VFUNC_CLEAN(CHOICE, NEXT_STATE_IDX))))
                ENDDO            
                QZERO = RISK/(1.+INTEREST)*A_GRID(CHOICE)
            ELSE
                DO NEXT_STATE_IDX=1, NZ
                    RISK = RISK + EMP_MARKOV(SCOLIDX, NEXT_STATE_IDX)*DFUNC(CHOICE, NEXT_STATE_IDX)
                ENDDO
                QZERO = (1.-RISK)/(1.+INTEREST)*A_GRID(CHOICE)
            ENDIF
            DIFF = Q(CHOICE, SCOLIDX) - QZERO
            ERROR_ARR(CHOICE, SCOLIDX) = ABS(DIFF)
            Q(CHOICE, SCOLIDX) = QZERO
        ENDDO  ! END SCOLIDX LOOP
    ENDDO
    ERROR_Q = MAXVAL(ABS(ERROR_ARR))

END SUBROUTINE

! SUBROUTINE EGM
! END SUBROUTINE

! SUBROUTINE Q_DERIV(CURRENTIDX)  ! GIVEN B' AND Y
!     USE ARRELANO_PARA
!     USE ARRELANO_RES
!     IMPLICIT NONE
!     REAL(KIND=8) DERIV  ! RESULT
!     INTEGER:: CHOICEIDX

!     DERIV = 0
!     DO CHOICEIDX = 1, NZ
!         DERIV = DERIV + CONT_PROB(B_PRIME, CHOICEIDX)*(1-ALPHA*(1-CONT_PROB(B_PRIME, CHOICEIDX))*c_prime)
!     ENDDO
! END SUBROUTINE

SUBROUTINE WRITE_ALL()
    USE ARRELANO_PARA
    USE ARRELANO_RES
    IMPLICIT NONE
    INTEGER:: SROWIDX
    INTEGER, DIMENSION(4):: SIDX
    CHARACTER(LEN=130):: PATH="./"
    CHARACTER(LEN=150):: FILE_NAME

        FILE_NAME = TRIM(PATH)//"VFUNC"
        OPEN(UNIT=1, FILE=FILE_NAME, STATUS='REPLACE') ! START WITH THE TWO VALUE FUNCTIONS
        DO SROWIDX=1, NA
            WRITE(UNIT=1,FMT=*) VFUNC_CLEAN(SROWIDX,:)
        ENDDO
        CLOSE(UNIT=1)

        ! FILE_NAME = TRIM(PATH)//"PFUNC"
        ! OPEN(UNIT=2, FILE=FILE_NAME, STATUS='REPLACE') ! ALSO SAVE POLICY FUNCTIONS
        ! DO SROWIDX=1, NA
        !     WRITE(UNIT=2,FMT=*) PFUNC_CLEAN(SROWIDX,:)
        ! ENDDO
        ! CLOSE(UNIT=2)

        FILE_NAME = TRIM(PATH)//"AGRID"
        OPEN(UNIT=4, FILE=FILE_NAME, STATUS='REPLACE') ! FOR HAVING THE X-AXIS OF PLOT
        WRITE(UNIT=4,FMT=*) A_GRID
        CLOSE(UNIT=4)

        FILE_NAME = TRIM(PATH)//"VFUND"
        OPEN(UNIT=5, FILE=FILE_NAME, STATUS='REPLACE') ! START WITH THE TWO VALUE FUNCTIONS
        DO SROWIDX=1, NA
            WRITE(UNIT=5,FMT=*) VFUNC_DEF(:)
        ENDDO
        CLOSE(UNIT=5)

        FILE_NAME = TRIM(PATH)//"Q"
        OPEN(UNIT=7, FILE=FILE_NAME, STATUS='REPLACE') ! ALSO SAVE POLICY FUNCTIONS
        DO SROWIDX=1, NA
            WRITE(UNIT=7,FMT=*) Q(SROWIDX,:)
        ENDDO
        CLOSE(UNIT=7)
END SUBROUTINE

SUBROUTINE CAL_MOMENTS()
    USE ARRELANO_PARA
    USE ARRELANO_RES
    IMPLICIT NONE
    REAL(KIND=8):: DEFAULT_HIGH=0., DEFAULT_LOW=0.
    INTEGER:: RIDX, CIDX, HIDX

    PRINT*, "AVG BOND PRICE", SUM(Q)/SIZE(Q)
END SUBROUTINE