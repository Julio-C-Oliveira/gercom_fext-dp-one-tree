# Executar a partir do diretório raiz: gercom_fext-dp-one-tree/

BASE_DIR="graphics/shap_analysis"

# 1. Cria as pastas de destino diretas
mkdir -p "${BASE_DIR}/waterfall_plots" "${BASE_DIR}/summary_plots"

# 2. Loop para copiar e renomear os arquivos
for eps in "eps_0.1" "eps_-1.0"; do
  # Extrai apenas o número do epsilon (ex: '0.1' ou '-1.0')
  val="${eps#eps_}"
  
  for strat in "ensemble_all_trees" "ensemble_threshold_trees" "merge_all_trees" "merge_threshold_trees"; do
    src="${BASE_DIR}/${strat}/${eps}/seed_42"
    dst_wf="${BASE_DIR}/waterfall_plots"
    dst_sum="${BASE_DIR}/summary_plots"
    
    # --- WATERFALL PLOTS ---
    cp "${src}/LOCAL_waterfall_single.pdf" "${dst_wf}/local_${val}_waterfall_single.pdf"
    cp "${src}/GLOBAL_waterfall_single.pdf" "${dst_wf}/${strat}_${val}_waterfall_single.pdf"
    
    # --- SUMMARY PLOTS ---
    cp "${src}/LOCAL_summary_global.pdf" "${dst_sum}/local_${val}_summary_global.pdf"
    cp "${src}/GLOBAL_summary_global.pdf" "${dst_sum}/${strat}_${val}_summary_global.pdf"
  done
done