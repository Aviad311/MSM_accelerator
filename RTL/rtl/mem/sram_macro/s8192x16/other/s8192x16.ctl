/* ctl_memcomp Version: 4.0.5-EAC4 */
/* common_memcomp Version: 4.0.5-EAC */
/* lang compiler Version: 4.0.0-beta23.4 Jun 21 2011 12:31:24 */
//
//       CONFIDENTIAL AND PROPRIETARY SOFTWARE OF ARM PHYSICAL IP, INC.
//      
//       Copyright (c) 1993 - 2026 ARM Physical IP, Inc.  All Rights Reserved.
//      
//       Use of this Software is subject to the terms and conditions of the
//       applicable license agreement with ARM Physical IP, Inc.
//       In addition, this Software is protected by patents, copyright law 
//       and international treaties.
//      
//       The copyright notice(s) in this Software does not indicate actual or
//       intended publication of this Software.
//
//      CTL model for Synchronous Single-Port Ram
//
//       Instance Name:              s8192x16
//       Words:                      8192
//       Bits:                       16
//       Mux:                        16
//       Drive:                      6
//       Write Mask:                 Off
//       Write Thru:                 Off
//       Extra Margin Adjustment:    On
//       Redundant Columns:          0
//       Test Muxes                  Off
//       Power Gating:               Off
//       Retention:                  On
//       Pipeline:                   Off
//       Read Disturb Test:	        Off
//       
//       Creation Date:  Sun Jun 28 00:22:45 2026
//       Version: 	r1p0
STIL 1.0 {
   CTL P2001.10;
   Design P2001.01;
}
Header {
   Title "CTL model for `s8192x16";
}
Signals {
   "Q[15]" Out;
   "Q[14]" Out;
   "Q[13]" Out;
   "Q[12]" Out;
   "Q[11]" Out;
   "Q[10]" Out;
   "Q[9]" Out;
   "Q[8]" Out;
   "Q[7]" Out;
   "Q[6]" Out;
   "Q[5]" Out;
   "Q[4]" Out;
   "Q[3]" Out;
   "Q[2]" Out;
   "Q[1]" Out;
   "Q[0]" Out;
   "CLK" In;
   "CEN" In;
   "WEN" In;
   "A[12]" In;
   "A[11]" In;
   "A[10]" In;
   "A[9]" In;
   "A[8]" In;
   "A[7]" In;
   "A[6]" In;
   "A[5]" In;
   "A[4]" In;
   "A[3]" In;
   "A[2]" In;
   "A[1]" In;
   "A[0]" In;
   "D[15]" In;
   "D[14]" In;
   "D[13]" In;
   "D[12]" In;
   "D[11]" In;
   "D[10]" In;
   "D[9]" In;
   "D[8]" In;
   "D[7]" In;
   "D[6]" In;
   "D[5]" In;
   "D[4]" In;
   "D[3]" In;
   "D[2]" In;
   "D[1]" In;
   "D[0]" In;
   "EMA[2]" In;
   "EMA[1]" In;
   "EMA[0]" In;
   "EMAW[1]" In;
   "EMAW[0]" In;
   "EMAS" In;
   "RET1N" In;
}
SignalGroups {
   "all_inputs" = '"CLK" + "CEN" + "WEN" + "A[12]" + "A[11]" + "A[10]" + "A[9]" + 
   "A[8]" + "A[7]" + "A[6]" + "A[5]" + "A[4]" + "A[3]" + "A[2]" + "A[1]" + "A[0]" + 
   "D[15]" + "D[14]" + "D[13]" + "D[12]" + "D[11]" + "D[10]" + "D[9]" + "D[8]" + 
   "D[7]" + "D[6]" + "D[5]" + "D[4]" + "D[3]" + "D[2]" + "D[1]" + "D[0]" + "EMA[2]" + 
   "EMA[1]" + "EMA[0]" + "EMAW[1]" + "EMAW[0]" + "EMAS" + "RET1N"';
   "all_outputs" = '"Q[15]" + "Q[14]" + "Q[13]" + "Q[12]" + "Q[11]" + "Q[10]" + "Q[9]" + 
   "Q[8]" + "Q[7]" + "Q[6]" + "Q[5]" + "Q[4]" + "Q[3]" + "Q[2]" + "Q[1]" + "Q[0]"';
   "all_ports" = '"all_inputs" + "all_outputs"';
   "_pi" = '"CLK" + "CEN" + "WEN" + "A[12]" + "A[11]" + "A[10]" + "A[9]" + "A[8]" + 
   "A[7]" + "A[6]" + "A[5]" + "A[4]" + "A[3]" + "A[2]" + "A[1]" + "A[0]" + "D[15]" + 
   "D[14]" + "D[13]" + "D[12]" + "D[11]" + "D[10]" + "D[9]" + "D[8]" + "D[7]" + "D[6]" + 
   "D[5]" + "D[4]" + "D[3]" + "D[2]" + "D[1]" + "D[0]" + "EMA[2]" + "EMA[1]" + "EMA[0]" + 
   "EMAW[1]" + "EMAW[0]" + "EMAS" + "RET1N"';
   "_po" = '"Q[15]" + "Q[14]" + "Q[13]" + "Q[12]" + "Q[11]" + "Q[10]" + "Q[9]" + 
   "Q[8]" + "Q[7]" + "Q[6]" + "Q[5]" + "Q[4]" + "Q[3]" + "Q[2]" + "Q[1]" + "Q[0]"';
}
ScanStructures {
}
Timing {
   WaveformTable "_default_WFT_" {
      Period '100ns';
      Waveforms {
         "all_inputs" {
            01ZN { '0ns' D/U/Z/N; }
         }
         "all_outputs" {
            XHTL { '40ns' X/H/T/L; }
         }
         "CLK" {
            P { '0ns' D; '45ns' U; '55ns' D; }
         }
      }
   }
}
Procedures {
   "capture" {
      W "_default_WFT_";
      V { "_pi" = #; "_po" = #; }
   }
   "capture_CLK" {
      W "_default_WFT_";
      V {"_pi" = #; "_po" = #;"CLK" = P; }
   }
   "load_unload" {
      W "_default_WFT_";
      V { "CLK" = 0; }
      Shift {
         V { "CLK" = P; }
      }
   }
}
MacroDefs {
   "test_setup" {
      W "_default_WFT_";
      C {"all_inputs" = \r60 N; "all_outputs" = \r34 X; }
      V { "CLK" = P; }
   }
}
Environment "s8192x16" {
   CTL {
   }
   CTL Internal_scan {
      TestMode InternalTest;
      Focus Top {
      }
      Internal {
      }
   }
}
Environment dftSpec {
   CTL {
   }
   CTL all_dft {
      TestMode ForInheritOnly;
   }
}
