/* ctl_memcomp Version: 4.0.5-EAC4 */
/* common_memcomp Version: 4.0.5-EAC */
/* lang compiler Version: 4.1.6-EAC2 Oct 30 2012 16:32:37 */
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
//       Instance Name:              s8192x64
//       Words:                      8192
//       Bits:                       64
//       Mux:                        16
//       Drive:                      6
//       Write Mask:                 Off
//       Write Thru:                 Off
//       Extra Margin Adjustment:    On
//       Redundant Columns:          2
//       Test Muxes                  On
//       Power Gating:               Off
//       Retention:                  On
//       Pipeline:                   Off
//       Read Disturb Test:	        Off
//       
//       Creation Date:  Fri Jun 26 13:36:51 2026
//       Version: 	r0p0
STIL 1.0 {
   CTL P2001.10;
   Design P2001.01;
}
Header {
   Title "CTL model for `s8192x64";
}
Signals {
   "CENY" Out;
   "WENY" Out;
   "AY[12]" Out;
   "AY[11]" Out;
   "AY[10]" Out;
   "AY[9]" Out;
   "AY[8]" Out;
   "AY[7]" Out;
   "AY[6]" Out;
   "AY[5]" Out;
   "AY[4]" Out;
   "AY[3]" Out;
   "AY[2]" Out;
   "AY[1]" Out;
   "AY[0]" Out;
   "Q[63]" Out;
   "Q[62]" Out;
   "Q[61]" Out;
   "Q[60]" Out;
   "Q[59]" Out;
   "Q[58]" Out;
   "Q[57]" Out;
   "Q[56]" Out;
   "Q[55]" Out;
   "Q[54]" Out;
   "Q[53]" Out;
   "Q[52]" Out;
   "Q[51]" Out;
   "Q[50]" Out;
   "Q[49]" Out;
   "Q[48]" Out;
   "Q[47]" Out;
   "Q[46]" Out;
   "Q[45]" Out;
   "Q[44]" Out;
   "Q[43]" Out;
   "Q[42]" Out;
   "Q[41]" Out;
   "Q[40]" Out;
   "Q[39]" Out;
   "Q[38]" Out;
   "Q[37]" Out;
   "Q[36]" Out;
   "Q[35]" Out;
   "Q[34]" Out;
   "Q[33]" Out;
   "Q[32]" Out;
   "Q[31]" Out;
   "Q[30]" Out;
   "Q[29]" Out;
   "Q[28]" Out;
   "Q[27]" Out;
   "Q[26]" Out;
   "Q[25]" Out;
   "Q[24]" Out;
   "Q[23]" Out;
   "Q[22]" Out;
   "Q[21]" Out;
   "Q[20]" Out;
   "Q[19]" Out;
   "Q[18]" Out;
   "Q[17]" Out;
   "Q[16]" Out;
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
   "SO[1]" Out;
   "SO[0]" Out;
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
   "D[63]" In;
   "D[62]" In;
   "D[61]" In;
   "D[60]" In;
   "D[59]" In;
   "D[58]" In;
   "D[57]" In;
   "D[56]" In;
   "D[55]" In;
   "D[54]" In;
   "D[53]" In;
   "D[52]" In;
   "D[51]" In;
   "D[50]" In;
   "D[49]" In;
   "D[48]" In;
   "D[47]" In;
   "D[46]" In;
   "D[45]" In;
   "D[44]" In;
   "D[43]" In;
   "D[42]" In;
   "D[41]" In;
   "D[40]" In;
   "D[39]" In;
   "D[38]" In;
   "D[37]" In;
   "D[36]" In;
   "D[35]" In;
   "D[34]" In;
   "D[33]" In;
   "D[32]" In;
   "D[31]" In;
   "D[30]" In;
   "D[29]" In;
   "D[28]" In;
   "D[27]" In;
   "D[26]" In;
   "D[25]" In;
   "D[24]" In;
   "D[23]" In;
   "D[22]" In;
   "D[21]" In;
   "D[20]" In;
   "D[19]" In;
   "D[18]" In;
   "D[17]" In;
   "D[16]" In;
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
   "TEN" In;
   "TCEN" In;
   "TWEN" In;
   "TA[12]" In;
   "TA[11]" In;
   "TA[10]" In;
   "TA[9]" In;
   "TA[8]" In;
   "TA[7]" In;
   "TA[6]" In;
   "TA[5]" In;
   "TA[4]" In;
   "TA[3]" In;
   "TA[2]" In;
   "TA[1]" In;
   "TA[0]" In;
   "TD[63]" In;
   "TD[62]" In;
   "TD[61]" In;
   "TD[60]" In;
   "TD[59]" In;
   "TD[58]" In;
   "TD[57]" In;
   "TD[56]" In;
   "TD[55]" In;
   "TD[54]" In;
   "TD[53]" In;
   "TD[52]" In;
   "TD[51]" In;
   "TD[50]" In;
   "TD[49]" In;
   "TD[48]" In;
   "TD[47]" In;
   "TD[46]" In;
   "TD[45]" In;
   "TD[44]" In;
   "TD[43]" In;
   "TD[42]" In;
   "TD[41]" In;
   "TD[40]" In;
   "TD[39]" In;
   "TD[38]" In;
   "TD[37]" In;
   "TD[36]" In;
   "TD[35]" In;
   "TD[34]" In;
   "TD[33]" In;
   "TD[32]" In;
   "TD[31]" In;
   "TD[30]" In;
   "TD[29]" In;
   "TD[28]" In;
   "TD[27]" In;
   "TD[26]" In;
   "TD[25]" In;
   "TD[24]" In;
   "TD[23]" In;
   "TD[22]" In;
   "TD[21]" In;
   "TD[20]" In;
   "TD[19]" In;
   "TD[18]" In;
   "TD[17]" In;
   "TD[16]" In;
   "TD[15]" In;
   "TD[14]" In;
   "TD[13]" In;
   "TD[12]" In;
   "TD[11]" In;
   "TD[10]" In;
   "TD[9]" In;
   "TD[8]" In;
   "TD[7]" In;
   "TD[6]" In;
   "TD[5]" In;
   "TD[4]" In;
   "TD[3]" In;
   "TD[2]" In;
   "TD[1]" In;
   "TD[0]" In;
   "RET1N" In;
   "SI[1]" In;
   "SI[0]" In;
   "SE" In;
   "DFTRAMBYP" In;
}
SignalGroups {
   "all_inputs" = '"CLK" + "CEN" + "WEN" + "A[12]" + "A[11]" + "A[10]" + "A[9]" + 
   "A[8]" + "A[7]" + "A[6]" + "A[5]" + "A[4]" + "A[3]" + "A[2]" + "A[1]" + "A[0]" + 
   "D[63]" + "D[62]" + "D[61]" + "D[60]" + "D[59]" + "D[58]" + "D[57]" + "D[56]" + 
   "D[55]" + "D[54]" + "D[53]" + "D[52]" + "D[51]" + "D[50]" + "D[49]" + "D[48]" + 
   "D[47]" + "D[46]" + "D[45]" + "D[44]" + "D[43]" + "D[42]" + "D[41]" + "D[40]" + 
   "D[39]" + "D[38]" + "D[37]" + "D[36]" + "D[35]" + "D[34]" + "D[33]" + "D[32]" + 
   "D[31]" + "D[30]" + "D[29]" + "D[28]" + "D[27]" + "D[26]" + "D[25]" + "D[24]" + 
   "D[23]" + "D[22]" + "D[21]" + "D[20]" + "D[19]" + "D[18]" + "D[17]" + "D[16]" + 
   "D[15]" + "D[14]" + "D[13]" + "D[12]" + "D[11]" + "D[10]" + "D[9]" + "D[8]" + 
   "D[7]" + "D[6]" + "D[5]" + "D[4]" + "D[3]" + "D[2]" + "D[1]" + "D[0]" + "EMA[2]" + 
   "EMA[1]" + "EMA[0]" + "EMAW[1]" + "EMAW[0]" + "TEN" + "TCEN" + "TWEN" + "TA[12]" + 
   "TA[11]" + "TA[10]" + "TA[9]" + "TA[8]" + "TA[7]" + "TA[6]" + "TA[5]" + "TA[4]" + 
   "TA[3]" + "TA[2]" + "TA[1]" + "TA[0]" + "TD[63]" + "TD[62]" + "TD[61]" + "TD[60]" + 
   "TD[59]" + "TD[58]" + "TD[57]" + "TD[56]" + "TD[55]" + "TD[54]" + "TD[53]" + "TD[52]" + 
   "TD[51]" + "TD[50]" + "TD[49]" + "TD[48]" + "TD[47]" + "TD[46]" + "TD[45]" + "TD[44]" + 
   "TD[43]" + "TD[42]" + "TD[41]" + "TD[40]" + "TD[39]" + "TD[38]" + "TD[37]" + "TD[36]" + 
   "TD[35]" + "TD[34]" + "TD[33]" + "TD[32]" + "TD[31]" + "TD[30]" + "TD[29]" + "TD[28]" + 
   "TD[27]" + "TD[26]" + "TD[25]" + "TD[24]" + "TD[23]" + "TD[22]" + "TD[21]" + "TD[20]" + 
   "TD[19]" + "TD[18]" + "TD[17]" + "TD[16]" + "TD[15]" + "TD[14]" + "TD[13]" + "TD[12]" + 
   "TD[11]" + "TD[10]" + "TD[9]" + "TD[8]" + "TD[7]" + "TD[6]" + "TD[5]" + "TD[4]" + 
   "TD[3]" + "TD[2]" + "TD[1]" + "TD[0]" + "RET1N" + "SI[1]" + "SI[0]" + "SE" + "DFTRAMBYP"';
   "all_outputs" = '"CENY" + "WENY" + "AY[12]" + "AY[11]" + "AY[10]" + "AY[9]" + 
   "AY[8]" + "AY[7]" + "AY[6]" + "AY[5]" + "AY[4]" + "AY[3]" + "AY[2]" + "AY[1]" + 
   "AY[0]" + "Q[63]" + "Q[62]" + "Q[61]" + "Q[60]" + "Q[59]" + "Q[58]" + "Q[57]" + 
   "Q[56]" + "Q[55]" + "Q[54]" + "Q[53]" + "Q[52]" + "Q[51]" + "Q[50]" + "Q[49]" + 
   "Q[48]" + "Q[47]" + "Q[46]" + "Q[45]" + "Q[44]" + "Q[43]" + "Q[42]" + "Q[41]" + 
   "Q[40]" + "Q[39]" + "Q[38]" + "Q[37]" + "Q[36]" + "Q[35]" + "Q[34]" + "Q[33]" + 
   "Q[32]" + "Q[31]" + "Q[30]" + "Q[29]" + "Q[28]" + "Q[27]" + "Q[26]" + "Q[25]" + 
   "Q[24]" + "Q[23]" + "Q[22]" + "Q[21]" + "Q[20]" + "Q[19]" + "Q[18]" + "Q[17]" + 
   "Q[16]" + "Q[15]" + "Q[14]" + "Q[13]" + "Q[12]" + "Q[11]" + "Q[10]" + "Q[9]" + 
   "Q[8]" + "Q[7]" + "Q[6]" + "Q[5]" + "Q[4]" + "Q[3]" + "Q[2]" + "Q[1]" + "Q[0]" + 
   "SO[1]" + "SO[0]"';
   "all_ports" = '"all_inputs" + "all_outputs"';
   "_pi" = '"CLK" + "CEN" + "WEN" + "A[12]" + "A[11]" + "A[10]" + "A[9]" + "A[8]" + 
   "A[7]" + "A[6]" + "A[5]" + "A[4]" + "A[3]" + "A[2]" + "A[1]" + "A[0]" + "D[63]" + 
   "D[62]" + "D[61]" + "D[60]" + "D[59]" + "D[58]" + "D[57]" + "D[56]" + "D[55]" + 
   "D[54]" + "D[53]" + "D[52]" + "D[51]" + "D[50]" + "D[49]" + "D[48]" + "D[47]" + 
   "D[46]" + "D[45]" + "D[44]" + "D[43]" + "D[42]" + "D[41]" + "D[40]" + "D[39]" + 
   "D[38]" + "D[37]" + "D[36]" + "D[35]" + "D[34]" + "D[33]" + "D[32]" + "D[31]" + 
   "D[30]" + "D[29]" + "D[28]" + "D[27]" + "D[26]" + "D[25]" + "D[24]" + "D[23]" + 
   "D[22]" + "D[21]" + "D[20]" + "D[19]" + "D[18]" + "D[17]" + "D[16]" + "D[15]" + 
   "D[14]" + "D[13]" + "D[12]" + "D[11]" + "D[10]" + "D[9]" + "D[8]" + "D[7]" + "D[6]" + 
   "D[5]" + "D[4]" + "D[3]" + "D[2]" + "D[1]" + "D[0]" + "EMA[2]" + "EMA[1]" + "EMA[0]" + 
   "EMAW[1]" + "EMAW[0]" + "TEN" + "TCEN" + "TWEN" + "TA[12]" + "TA[11]" + "TA[10]" + 
   "TA[9]" + "TA[8]" + "TA[7]" + "TA[6]" + "TA[5]" + "TA[4]" + "TA[3]" + "TA[2]" + 
   "TA[1]" + "TA[0]" + "TD[63]" + "TD[62]" + "TD[61]" + "TD[60]" + "TD[59]" + "TD[58]" + 
   "TD[57]" + "TD[56]" + "TD[55]" + "TD[54]" + "TD[53]" + "TD[52]" + "TD[51]" + "TD[50]" + 
   "TD[49]" + "TD[48]" + "TD[47]" + "TD[46]" + "TD[45]" + "TD[44]" + "TD[43]" + "TD[42]" + 
   "TD[41]" + "TD[40]" + "TD[39]" + "TD[38]" + "TD[37]" + "TD[36]" + "TD[35]" + "TD[34]" + 
   "TD[33]" + "TD[32]" + "TD[31]" + "TD[30]" + "TD[29]" + "TD[28]" + "TD[27]" + "TD[26]" + 
   "TD[25]" + "TD[24]" + "TD[23]" + "TD[22]" + "TD[21]" + "TD[20]" + "TD[19]" + "TD[18]" + 
   "TD[17]" + "TD[16]" + "TD[15]" + "TD[14]" + "TD[13]" + "TD[12]" + "TD[11]" + "TD[10]" + 
   "TD[9]" + "TD[8]" + "TD[7]" + "TD[6]" + "TD[5]" + "TD[4]" + "TD[3]" + "TD[2]" + 
   "TD[1]" + "TD[0]" + "RET1N" + "SI[1]" + "SI[0]" + "SE" + "DFTRAMBYP"';
   "_po" = '"CENY" + "WENY" + "AY[12]" + "AY[11]" + "AY[10]" + "AY[9]" + "AY[8]" + 
   "AY[7]" + "AY[6]" + "AY[5]" + "AY[4]" + "AY[3]" + "AY[2]" + "AY[1]" + "AY[0]" + 
   "Q[63]" + "Q[62]" + "Q[61]" + "Q[60]" + "Q[59]" + "Q[58]" + "Q[57]" + "Q[56]" + 
   "Q[55]" + "Q[54]" + "Q[53]" + "Q[52]" + "Q[51]" + "Q[50]" + "Q[49]" + "Q[48]" + 
   "Q[47]" + "Q[46]" + "Q[45]" + "Q[44]" + "Q[43]" + "Q[42]" + "Q[41]" + "Q[40]" + 
   "Q[39]" + "Q[38]" + "Q[37]" + "Q[36]" + "Q[35]" + "Q[34]" + "Q[33]" + "Q[32]" + 
   "Q[31]" + "Q[30]" + "Q[29]" + "Q[28]" + "Q[27]" + "Q[26]" + "Q[25]" + "Q[24]" + 
   "Q[23]" + "Q[22]" + "Q[21]" + "Q[20]" + "Q[19]" + "Q[18]" + "Q[17]" + "Q[16]" + 
   "Q[15]" + "Q[14]" + "Q[13]" + "Q[12]" + "Q[11]" + "Q[10]" + "Q[9]" + "Q[8]" + 
   "Q[7]" + "Q[6]" + "Q[5]" + "Q[4]" + "Q[3]" + "Q[2]" + "Q[1]" + "Q[0]" + "SO[1]" + 
   "SO[0]"';
   "_si" = '"SI[0]" + "SI[1]"' {ScanIn; }
   "_so" = '"SO[0]" + "SO[1]"' {ScanOut; }
}
ScanStructures {
   ScanChain "chain_s8192x64_1" {
      ScanLength  32;
      ScanCells   "uDQ0" "uDQ1" "uDQ2" "uDQ3" "uDQ4" "uDQ5" "uDQ6" "uDQ7" "uDQ8" "uDQ9" "uDQ10" "uDQ11" "uDQ12" "uDQ13" "uDQ14" "uDQ15" "uDQ16" "uDQ17" "uDQ18" "uDQ19" "uDQ20" "uDQ21" "uDQ22" "uDQ23" "uDQ24" "uDQ25" "uDQ26" "uDQ27" "uDQ28" "uDQ29" "uDQ30" "uDQ31" ;
      ScanIn  "SI[0]";
      ScanOut  "SO[0]";
      ScanEnable  "SE";
      ScanMasterClock  "CLK";
   }
   ScanChain "chain_s8192x64_2" {
      ScanLength  32;
      ScanCells  "uDQ63" "uDQ62" "uDQ61" "uDQ60" "uDQ59" "uDQ58" "uDQ57" "uDQ56" "uDQ55" "uDQ54" "uDQ53" "uDQ52" "uDQ51" "uDQ50" "uDQ49" "uDQ48" "uDQ47" "uDQ46" "uDQ45" "uDQ44" "uDQ43" "uDQ42" "uDQ41" "uDQ40" "uDQ39" "uDQ38" "uDQ37" "uDQ36" "uDQ35" "uDQ34" "uDQ33" "uDQ32"  ;
      ScanIn  "SI[1]";
      ScanOut  "SO[1]";
      ScanEnable  "SE";
      ScanMasterClock  "CLK";
   }
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
      V { "CLK" = 0; "_si" = \r2 N; "_so" =\r2 X; "SE" = 1; "DFTRAMBYP" = 1; }
      Shift {
         V { "CLK" = P; "_si" = \r2 #; "_so" = \r2 #; }
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
Environment "s8192x64" {
   CTL {
   }
   CTL Internal_scan {
      TestMode InternalTest;
      Focus Top {
      }
      Internal {
         "SI[0]" {
            CaptureClock "CLK" {
               LeadingEdge;
            }
            DataType ScanDataIn {
               ScanDataType Internal;
            }
            ScanStyle MultiplexedData;
         }
         "SI[1]" {
            CaptureClock "CLK" {
               LeadingEdge;
            }
            DataType ScanDataIn {
               ScanDataType Internal;
            }
            ScanStyle MultiplexedData;
         }
         "SO[0]" {
            LaunchClock "CLK" {
               LeadingEdge;
            }
            DataType ScanDataOut {
               ScanDataType Internal;
            }
            ScanStyle MultiplexedData;
         }
         "SO[1]" {
            LaunchClock "CLK" {
               LeadingEdge;
            }
            DataType ScanDataOut {
               ScanDataType Internal;
            }
            ScanStyle MultiplexedData;
         }
         "SE" {
            DataType ScanEnable {
               ActiveState ForceUp;
            }
         }
         "CLK" {
            DataType ScanMasterClock MasterClock;
         }
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
