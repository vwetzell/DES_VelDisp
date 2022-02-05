#!/usr/bin/env python

from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt
import sys 
import os
from astropy.table import Table, Column, vstack
import glob
import warnings

warnings.filterwarnings('ignore')

from bootstrap import *
from Beers import *
from tbl_reader import *


def VelDisp(t,clust_dict,data_clust_vlim,
    cluster_file_name,member_file_name,bootnum,cores=1):

    clust_vlim = set(data_clust_vlim['MEM_MATCH_ID'])
        
    
    N     = 0    
    n     = 0
    index = 0
    h     = 0
        
    l = Table(names=('ID','MEM_MATCH_ID','Z','PMEM','ZSPEC','R'))
    l_all = Table(names=('ID','MEM_MATCH_ID','Z','PMEM','ZSPEC','R'))
    l_all_interlopper = Table(names=('ID','MEM_MATCH_ID',
                                     'Z','PMEM','ZSPEC','R'))    

    tbl = Table(names=('MEM_MATCH_ID','RA','DEC',
                    'N','M','Z_LAMBDA','c_gal','vol_lim','outlier',
                    'c_bi',
                    'sigma_g','sigma_g_err_low','sigma_g_err_high',
                    'sigma_g2000','sigma_g2000_err_low','sigma_g2000_err_high',
                    'sigma_g5','sigma_g5_err_low','sigma_g5_err_high',
                    'sigma_g8','sigma_g8_err_low','sigma_g8_err_high',
                    'sigma_bi','sigma_bi_err_low','sigma_bi_err_high',
                    'sigma_bi5','sigma_bi5_err_low','sigma_bi5_err_high',
                    'sigma_bi8','sigma_bi8_err_low','sigma_bi8_err_high',
                    'LAMBDA_CHISQ','lambda_err'))

    plt.figure()
    f, axarr = plt.subplots(42,6,figsize=(20,100))
    z = np.linspace(-3000,3000,1000)

    for i in t['ID']:
        if t[i]['MEM_MATCH_ID']==n:
            l.add_row((index,
                    t['MEM_MATCH_ID'][i],
                    t['Z'][i],
                    t['PMEM'][i],
                    t['ZSPEC'][i],
                    t['R'][i]
                    ))
            continue
            
        if len(l['MEM_MATCH_ID']) >= 15:
            
            c_bi = center_cluster_biweight(np.array(l['ZSPEC']))

            c_gal = np.nan
            if 1.0 in l['PMEM']:
                c_gal = l['ZSPEC'][l['PMEM']==1.0]
                
            v_bi  = z2v(np.array(l['ZSPEC']),c_bi)
            v_gal = z2v(np.array(l['ZSPEC']),c_gal)
            
            v_bi_col  = Column(name='v_bi',data=v_bi)
            v_gal_col = Column(name='v_gal',data=v_gal)
            
            l.add_column(v_bi_col)
            l.add_column(v_gal_col)
            
            photoz        = clust_dict[n][2]
            clust_lam     = clust_dict[n][4]
            clust_lam_err = clust_dict[n][3]
            clust_ra      = clust_dict[n][0]
            clust_dec     = clust_dict[n][1]
            
            vlim = False
            if n in clust_vlim:
                vlim = True

                
            member      = abs(v_bi)<3000*(clust_lam/20)**0.45
            member2000  = abs(v_bi)<2000*(clust_lam/20)**0.45
            member5     = np.logical_and(member,l['PMEM']>0.5)
            member8     = np.logical_and(member,l['PMEM']>0.8)
            
            member_col  = Column(name='member',data = member)
            member2000_col  = Column(name='member2000',data = member2000)
            member5_col = Column(name='member5',data = member5)
            member8_col = Column(name='member8',data = member8)
            
            lambda_col = Column(name='LAMBDA',
                data=np.full(len(l['MEM_MATCH_ID']),clust_lam))
            
            l.add_column(member_col)
            l.add_column(member2000_col)
            l.add_column(member5_col)
            l.add_column(member8_col)
            l.add_column(lambda_col)
            
            N = len(l['MEM_MATCH_ID'][member])
            
        if N >= 15:
            
            # No PMEM Cut
            v_bi_temp = v_bi[member]
            
            boot_bi = bootstrap(v_bi_temp,
                                bootnum=bootnum,bootfunc=sigma_bi,cores=cores)
            boot_bi = boot_bi[~np.isnan(boot_bi)]
            boot_bi = np.sort(boot_bi)
            
            sig_bi      = np.median(boot_bi)
            sig_bi_low  = sig_bi - boot_bi[int(0.16*len(boot_bi))]
            sig_bi_high = boot_bi[int(0.84*len(boot_bi))] - sig_bi
            
            boot_g = bootstrap(v_bi_temp,
                            bootnum=bootnum,bootfunc=gapper,cores=cores)
            boot_g = boot_g[~np.isnan(boot_g)]
            boot_g = np.sort(boot_g)
            
            sig_g      = np.median(boot_g)
            sig_g_low  = sig_g - boot_g[int(0.16*len(boot_g))]
            sig_g_high = boot_g[int(0.84*len(boot_g))] - sig_g
            
            # 2000 km/s cut
            sig_bi2000      = np.nan
            sig_bi_low2000  = np.nan
            sig_bi_high2000 = np.nan
            sig_g2000      = np.nan
            sig_g_low2000  = np.nan
            sig_g_high2000 = np.nan
            if sum(member2000)>=15:
                v_bi_temp = v_bi[member2000]
                
                boot_bi2000 = bootstrap(v_bi_temp,
                                    bootnum=bootnum,
                                    bootfunc=sigma_bi,cores=cores)
                boot_bi2000 = boot_bi2000[~np.isnan(boot_bi2000)]
                boot_bi2000 = np.sort(boot_bi2000)
                
                sig_bi2000      = np.median(boot_bi2000)
                sig_bi_low2000  = (sig_bi2000 
                    - boot_bi2000[int(0.16*len(boot_bi2000))])
                sig_bi_high2000 = (boot_bi2000[int(0.84*len(boot_bi2000))] 
                    - sig_bi2000)
                
                boot_g2000 = bootstrap(v_bi_temp,
                                bootnum=bootnum,
                                bootfunc=gapper,cores=cores)
                boot_g2000 = boot_g2000[~np.isnan(boot_g2000)]
                boot_g2000 = np.sort(boot_g2000)
                
                sig_g2000      = np.median(boot_g2000)
                sig_g_low2000  = (sig_g2000 
                    - boot_g2000[int(0.16*len(boot_g2000))])
                sig_g_high2000 = (boot_g2000[int(0.84*len(boot_g2000))] 
                    - sig_g2000)
            
            
            # PMEM > 0.50
            sig_bi5      = np.nan
            sig_bi5_low  = np.nan
            sig_bi5_high = np.nan
            sig_g5      = np.nan
            sig_g5_low  = np.nan
            sig_g5_high = np.nan
            if sum(member5)>=15:
                v_bi_temp = v_bi
                
                boot_bi5 = bootstrap(v_bi_temp[member5],
                                    bootnum=bootnum,
                                    bootfunc=sigma_bi,cores=cores)
                boot_bi5 = boot_bi5[~np.isnan(boot_bi5)]
                boot_bi5 = np.sort(boot_bi5)
                
                sig_bi5      = np.median(boot_bi5)
                sig_bi5_low  = sig_bi5 - boot_bi5[int(0.16*len(boot_bi5))]
                sig_bi5_high = boot_bi5[int(0.84*len(boot_bi5))] - sig_bi5
                
                boot_g5 = bootstrap(v_bi_temp[member5],
                                bootnum=bootnum,
                                bootfunc=gapper,cores=cores)
                boot_g5 = boot_g5[~np.isnan(boot_g5)]
                boot_g5 = np.sort(boot_g5)
                
                sig_g5      = np.median(boot_g5)
                sig_g5_low  = sig_g5 - boot_g5[int(0.16*len(boot_g5))]
                sig_g5_high = boot_g5[int(0.84*len(boot_g5))] - sig_g5
            
            
            # PMEM > 0.80
            sig_bi8      = np.nan
            sig_bi8_low  = np.nan
            sig_bi8_high = np.nan
            sig_g8      = np.nan
            sig_g8_low  = np.nan
            sig_g8_high = np.nan
            if sum(member8)>=15:
                v_bi_temp = v_bi
                
                boot_bi8 = bootstrap(v_bi_temp[member8],
                                    bootnum=bootnum,
                                    bootfunc=sigma_bi,cores=cores)
                boot_bi8 = boot_bi8[~np.isnan(boot_bi8)]
                boot_bi8 = np.sort(boot_bi8)
                
                sig_bi8      = np.median(boot_bi8)
                sig_bi8_low  = sig_bi8 - boot_bi8[int(0.16*len(boot_bi8))]
                sig_bi8_high = boot_bi8[int(0.84*len(boot_bi8))] - sig_bi8
                
                boot_g8 = bootstrap(v_bi_temp[member8],
                                bootnum=bootnum,
                                bootfunc=gapper,cores=cores)
                boot_g8 = boot_g8[~np.isnan(boot_g8)]
                boot_g8 = np.sort(boot_g8)
                
                sig_g8      = np.median(boot_g8)
                sig_g8_low  = sig_g8 - boot_g8[int(0.16*len(boot_g8))]
                sig_g8_high = boot_g8[int(0.84*len(boot_g8))] - sig_g8
            
            
            sig_g_mem = np.logical_and(member,
                abs(v_bi)<3*sig_g*(clust_lam/20)**0.45)
            sig_g_mem_col  = Column(name='sig_g_mem',data = sig_g_mem)
            l.add_column(sig_g_mem_col)
            
            outlier=False
            
            N = len(l['MEM_MATCH_ID'][sig_g_mem])
            M = len(l['MEM_MATCH_ID'])-N
            
            c = h//6
            r = h%6
            
            axarr[2*c+1,r].hist(boot_g,
                histtype='step',color='b',range=(0,3000),bins=20)
            axarr[2*c+1,r].axvspan(sig_g-sig_g_low,sig_g_high+sig_g,
                color='gray',alpha=0.2,lw=2)
            axarr[2*c+1,r].axvline(sig_g,c='k')
            axarr[2*c+1,r].set_xlim(0,3000)
            axarr[2*c+1,r].set_xticks([0,1000,2000,3000])
            axarr[2*c+1,r].set_xticklabels([0,1000,2000,3000],fontsize=15)
            yticks = axarr[2*c+1,r].get_yticks()
            axarr[2*c+1,r].set_yticks(yticks)
            axarr[2*c+1,r].set_yticklabels([int(i) for i in yticks],
                fontsize=15)
            axarr[2*c+1,r].set_xlabel(r'$\sigma_G$ (km/s)',fontsize=20)
            
            axarr[2*c,r].hist((v_bi[sig_g_mem],v_bi[~sig_g_mem]),
                bins=20,range=(-5000,5000),color=('b','r'),
                histtype='bar',rwidth=0.7,stacked=True)
            axarr[2*c,r].set_xlim(-5000,5000)
            axarr[2*c,r].set_ylim(0,20)
            axarr[2*c,r].set_xticks([-3000,0,3000])
            axarr[2*c,r].set_xticklabels([-3000,0,3000],fontsize=15)
            axarr[2*c,r].set_yticks([0,5,10,15,20])
            axarr[2*c,r].set_yticklabels([0,5,10,15,20],fontsize=15)
            axarr[2*c,r].set_xlabel(r'$v_{pec}$ (km/s)',fontsize=20)
            if ((sig_g - sig_g_low)/((1+c_bi)/(1+0.171))**0.54
                >(618.1+240)*(clust_lam/(33.336*0.92))**0.435):
                axarr[2*c,r].text(x=-4700,y=16.5,s=' {} *'.format(str(int(n))),
                        color='r',fontsize=20)
                outlier=True
            else:
                axarr[2*c,r].text(x=-4700,y=16.5,s=' {}'.format(str(int(n))),
                        fontsize=20)
            
            l_all = vstack([l_all,l])
            
            tbl.add_row((n,clust_ra,clust_dec,N,M,photoz,c_gal,vlim,outlier,
                        c_bi,
                        sig_g,sig_g_low,sig_g_high,
                        sig_g2000,sig_g_low2000,sig_g_high2000,
                        sig_g5,sig_g5_low,sig_g5_high,
                        sig_g8,sig_g8_low,sig_g8_high,
                        sig_bi,sig_bi_low,sig_bi_high,
                        sig_bi5,sig_bi5_low,sig_bi5_high,
                        sig_bi8,sig_bi8_low,sig_bi8_high,
                        clust_lam,clust_lam_err))
            
            
            n=t[i]['MEM_MATCH_ID']
            N = 0
            h+=1
            index += 1
            l = Table(names=('ID','MEM_MATCH_ID','Z','PMEM','ZSPEC','R'))
            l.add_row((index,
                    t['MEM_MATCH_ID'][i],
                    t['Z'][i],
                    t['PMEM'][i],
                    t['ZSPEC'][i],
                    t['R'][i]
                    ))
            continue
            
        else:
            
            n=t[i]['MEM_MATCH_ID']
            N = 0
            l = Table(names=('ID','MEM_MATCH_ID','Z','PMEM','ZSPEC','R'))
            l.add_row((index,
                    t['MEM_MATCH_ID'][i],
                    t['Z'][i],
                    t['PMEM'][i],
                    t['ZSPEC'][i],
                    t['R'][i]
                    ))
    tbl_members = l_all
    f.tight_layout(w_pad=0.1,h_pad=0.3)
    plt.savefig('bootstrap_gallery.png',bbox_inches='tight',dpi=300)
    plt.close()

    tbl.write(cluster_file_name+'.fits',overwrite=True)
    tbl_members.write(member_file_name+'.fits',overwrite=True)

    return



if __name__=='__main__':
    help = "Still need to write the help section"

    my_cores=os.cpu_count()
    if len(sys.argv)==4:
        if sys.argv[1]=='-h' or sys.argv[1]=='--help':
            print(help)
            sys.exit(1)
        member_cat = sys.argv[1]
        cluster_cat = sys.argv[2]
        vlim_cat = sys.argv[3]
    else:
        print(help)
        sys.exit(1)

    print('Reading data tables...')

    data = read_tbl_data(member_cat)
    data_clust = read_tbl_data(cluster_cat)
    data_clust_vlim = read_tbl_data(vlim_cat)

    print('Data tables read...')

    t = Table()

    t['ID']                  = list(range(len(data['MEM_MATCH_ID'])))
    t['MEM_MATCH_ID']        = data['MEM_MATCH_ID']
    t['Z']                   = data['Z_1']
    t['PMEM']                = data['P']
    t['ZSPEC']               = data['ZSPEC']
    t['ZSPEC'][t['ZSPEC']<0] = data['Z_2'][t['ZSPEC']<0]
    t['R']                   = data['R']

    clust_dict = dict(zip(data_clust['MEM_MATCH_ID'],
                      zip(data_clust['RA'],
                          data_clust['DEC'],
                          data_clust['Z_LAMBDA'],
                          data_clust['LAMBDA_CHISQ_E'],
                          data_clust['LAMBDA_CHISQ'])
                      ))

    clust_vlim = set(data_clust_vlim['MEM_MATCH_ID'])

    print('Starting VelDisp...')

    VelDisp(t,clust_dict,data_clust_vlim,
        cluster_file_name='DES_VelDisp_Clusters',
        member_file_name='DES_VelDisp_Memebers',
        cores=my_cores,bootnum=10000)

    print('VelDisp complete! \
        Cluster data table => '+'DES_VelDisp_Clusters'+'.fits \
        Member data talble => '+'DES_VelDisp_Memebers'+'.fits \
        Gallery            => bootstrap_gallery.png')

    sys.exit(0)