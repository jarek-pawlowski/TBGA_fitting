dictionary = {"Subscript(V,p**2*\[Sigma])": "self.m.Vpp_sigma_inter",
              "Subscript(V,p**2*Pi)": "self.m.Vpp_pi_inter",
              "Subscript(V,d**2*\[Sigma])": "self.m.Vdd_sigma_inter", 
              "Subscript(V,d**2*Pi)": "self.m.Vdd_pi_inter",
              "Subscript(V,d**2*\[Delta])": "self.m.Vdd_delta_inter",
              "Subscript(V,dp\[Sigma])": "self.m.Vdp_sigma_inter",
              "Subscript(V,pd\[Sigma])": "self.m.Vpd_sigma_inter",
              "Subscript(V,dp\[Pi])": "self.m.Vdp_pi_inter",
              "Subscript(V,pd\[Pi])": "self.m.Vpd_pi_inter",
              "Sqrt": "np.sqrt",
              "E**": "np.exp",
              "(0,1)": "1.j",
              "dzdd": "dz_dd",
              "dzpp": "dz_pp",
              "dzdp": "dz_dp",
              "dzpd": "dz_pd",              
              "dr*": "d1*",
              "dr -": "d1 -",
              "dr-": "d1-",
              "dr +": "d1 +",
              "dr+": "d1+",
              "*dr": "*d1",
              "I": "1.j"}

processed = []
with open('interlayer.f') as f:
    lines = f.readlines() # list containing lines of file
    for line in lines:
        line = line.strip()
        for key in dictionary.keys():
            line = line.replace(key, dictionary[key])
        processed.append(line)

with open("interlayer.py", "w") as f:
    for line in processed:
        f.write(line +'\n')
f.close()