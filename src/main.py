"""
Streamlit web interface for Manuscript Journal Matcher.

Complete interface with Steps 2-5 functionality:
- Document Extraction (Step 2)
- Embedding System (Step 3) 
- Journal Database (Step 4)
- Vector Search (Step 5)
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import traceback
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    # Try absolute imports first
    from src.extractor import extract_manuscript_data, validate_extracted_data, ExtractionError
    from src.embedder import embed_text, cosine_similarity_single, get_embedding_info, EmbeddingError
    from src.match_journals import JournalMatcher, MatchingError, format_search_results
    from src.journal_db_builder import load_journal_database
except ImportError:
    # Fallback to direct imports
    from extractor import extract_manuscript_data, validate_extracted_data, ExtractionError
    from embedder import embed_text, cosine_similarity_single, get_embedding_info, EmbeddingError
    from match_journals import JournalMatcher, MatchingError, format_search_results
    from journal_db_builder import load_journal_database

st.set_page_config(
    page_title="Manuscript Journal Matcher",
    page_icon="📄",
    layout="wide"
)

def main():
    """Main Streamlit application."""
    st.title("📄 Manuscript Journal Matcher")
    st.markdown("**Complete System** - Steps 2-5 Implemented")
    
    # Initialize session state
    if 'matcher' not in st.session_state:
        st.session_state.matcher = None
    
    # Sidebar with system info
    with st.sidebar:
        st.header("System Status")
        
        # Initialize journal matcher
        if st.button("🔄 Initialize System"):
            initialize_system()
        
        # Show system status
        display_system_status()
        
        st.markdown("---")
        st.markdown("**Available Features:**")
        st.markdown("✅ Document text extraction")
        st.markdown("✅ Title & abstract detection")
        st.markdown("✅ Text embedding generation")
        st.markdown("✅ Journal database (7,648+ journals)")
        st.markdown("✅ Semantic journal matching")
        st.markdown("✅ Results ranking & filtering")
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["📄 Upload Manuscript", "🧪 Quick Test", "📊 Database Info"])
    
    with tab1:
        st.markdown("Upload a PDF or DOCX file to find matching journals")
        
        uploaded_file = st.file_uploader(
            "Choose a manuscript file",
            type=['pdf', 'docx'],
            help="Supported formats: PDF (.pdf) and Word Document (.docx)"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size:,} bytes")
            st.write(f"**Type:** {uploaded_file.type}")
            
            # Process button
            if st.button("🔍 Extract and Find Journals", type="primary"):
                process_manuscript(uploaded_file)
    
    with tab2:
        st.markdown("Test journal matching with sample abstract text")
        
        sample_abstracts = {
            "Medical Research": "This study investigates the effectiveness of machine learning algorithms in diagnosing cardiovascular diseases using ECG data from 10,000 patients.",
            "Computer Science": "We present a novel deep learning architecture for natural language processing tasks, achieving state-of-the-art performance on multiple benchmarks.",
            "Biology": "Our research examines the genetic mechanisms underlying cellular differentiation in stem cells using CRISPR-Cas9 gene editing techniques.",
            "Custom": ""  # For user input
        }
        
        selected_type = st.selectbox("Choose sample abstract or enter custom:", list(sample_abstracts.keys()))
        
        if selected_type == "Custom":
            abstract_text = st.text_area("Enter your abstract:", height=150)
        else:
            abstract_text = sample_abstracts[selected_type]
            st.text_area("Abstract:", value=abstract_text, height=100, disabled=True)
        
        if abstract_text and st.button("🔍 Find Matching Journals", type="primary"):
            find_matching_journals(abstract_text)
    
    with tab3:
        display_database_info()


def process_manuscript(uploaded_file):
    """Process uploaded manuscript file."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        with st.spinner("Extracting text from manuscript..."):
            # Extract manuscript data
            manuscript_data = extract_manuscript_data(tmp_file_path)
            
            # Validate extraction
            validation = validate_extracted_data(manuscript_data)
        
        # Clean up temp file
        Path(tmp_file_path).unlink()
        
        # Display results
        display_extraction_results(manuscript_data, validation)
        
        # Find matching journals if we have an abstract
        if manuscript_data.get('abstract'):
            st.markdown("---")
            find_matching_journals(manuscript_data['abstract'])
    
    except ExtractionError as e:
        st.error(f"❌ Extraction failed: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.code(traceback.format_exc())


def display_extraction_results(data, validation):
    """Display manuscript extraction results."""
    st.success("✅ Manuscript processed successfully!")
    
    # Show validation status
    if validation['status'] == 'valid':
        st.success("✅ Extraction validation passed")
    else:
        st.warning("⚠️ Extraction validation issues")
    
    if validation['warnings']:
        st.warning("**Warnings:**")
        for warning in validation['warnings']:
            st.write(f"- {warning}")
    
    # Display extracted content in tabs
    tab1, tab2, tab3 = st.tabs(["📋 Summary", "📝 Title", "📄 Abstract"])
    
    with tab1:
        st.write("**Extraction Summary:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Title Found", "✅" if data.get('title') else "❌")
            st.metric("Abstract Found", "✅" if data.get('abstract') else "❌")
        with col2:
            st.metric("File Type", data.get('file_type', 'Unknown'))
            st.metric("Text Length", f"{len(data.get('full_text', ''))}" + " chars")
    
    with tab2:
        st.write("**Extracted Title:**")
        if data.get('title'):
            st.info(data['title'])
        else:
            st.warning("No title found")
    
    with tab3:
        st.write("**Extracted Abstract:**")
        if data.get('abstract'):
            st.info(data['abstract'])
            st.write(f"**Length:** {len(data['abstract'])} characters")
        else:
            st.warning("No abstract found")


def generate_embeddings_demo(text):
    """Demonstrate embedding generation."""
    st.header("🧠 Embedding Analysis")
    
    with st.spinner("Generating embeddings..."):
        try:
            # Generate embedding
            embedding = embed_text(text[:500])  # Limit length for demo
            
            st.success("✅ Embedding generated successfully!")
            
            # Display embedding info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Embedding Dimension", embedding.shape[0])
            with col2:
                st.metric("Data Type", str(embedding.dtype))
            with col3:
                st.metric("Memory Size", f"{embedding.nbytes} bytes")
            
            # Show sample embedding values
            st.write("**Sample Embedding Values:**")
            st.code(f"[{', '.join([f'{x:.4f}' for x in embedding[:10]])}...]")
            
            # Self-similarity test
            self_similarity = cosine_similarity_single(embedding, embedding)
            st.metric("Self-similarity", f"{self_similarity:.6f}", help="Should be 1.000000")
            
            if abs(self_similarity - 1.0) < 1e-6:
                st.success("✅ Self-similarity test passed")
            else:
                st.error("❌ Self-similarity test failed")
        
        except EmbeddingError as e:
            st.error(f"❌ Embedding generation failed: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")


def demo_analysis():
    """Run demo analysis with sample text."""
    sample_text = """
    Advanced Machine Learning Approaches for Biomedical Data Analysis
    
    Abstract: This research presents a comprehensive investigation into the application 
    of machine learning algorithms for analyzing complex biomedical datasets. Our study 
    evaluates multiple algorithmic approaches including deep neural networks, random 
    forests, and support vector machines on various healthcare data types. The results 
    demonstrate significant improvements in diagnostic accuracy, with neural networks 
    achieving 94.2% accuracy compared to traditional methods at 87.3%. Additionally, 
    we introduce a novel feature selection technique that reduces computational complexity 
    by 35% while maintaining prediction performance.
    
    Keywords: machine learning, biomedical data, healthcare, neural networks
    
    1. Introduction
    The healthcare industry has experienced rapid technological advancement...
    """
    
    st.success("🧪 Running demo analysis...")
    
    with st.spinner("Processing sample text..."):
        try:
            # Extract title and abstract
            try:
                from src.extractor import extract_title_and_abstract
            except ImportError:
                from extractor import extract_title_and_abstract
            title, abstract = extract_title_and_abstract(sample_text)
            
            # Display results
            st.write("**Demo Results:**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Extracted Title:**")
                st.info(title if title else "No title found")
            
            with col2:
                st.write("**Abstract Found:**")
                st.success("✅ Yes" if abstract else "❌ No")
            
            if abstract:
                st.write("**Extracted Abstract:**")
                st.info(abstract)
                
                # Generate embedding
                st.write("**Embedding Generation:**")
                embedding = embed_text(abstract)
                st.success(f"✅ Generated {embedding.shape[0]}-dimensional embedding")
                
                # Test similarity with modified version
                modified_abstract = abstract.replace("machine learning", "artificial intelligence")
                modified_embedding = embed_text(modified_abstract)
                similarity = cosine_similarity_single(embedding, modified_embedding)
                
                st.write("**Similarity Test:**")
                st.info(f"Original vs Modified text similarity: {similarity:.4f}")
                
                if similarity > 0.8:
                    st.success("✅ High semantic similarity detected")
                else:
                    st.warning("⚠️ Low semantic similarity")
        
        except Exception as e:
            st.error(f"❌ Demo failed: {e}")


def initialize_system():
    """Initialize the journal matching system."""
    try:
        with st.spinner("Initializing journal matcher..."):
            matcher = JournalMatcher()
            matcher.load_database()
            st.session_state.matcher = matcher
            st.success("✅ System initialized successfully!")
    except Exception as e:
        st.error(f"❌ System initialization failed: {e}")
        st.session_state.matcher = None


def display_system_status():
    """Display current system status in sidebar."""
    # Embedding model info
    model_info = get_embedding_info()
    if model_info.get('loaded', False):
        st.success("✅ Embedding Model")
        st.write(f"**Model:** {model_info.get('model_name', 'Unknown')}")
        st.write(f"**Dimension:** {model_info.get('dimension', 'Unknown')}")
    else:
        st.error("❌ Embedding Model")
        if model_info.get('error'):
            st.write(f"Error: {model_info['error']}")
    
    # Journal database status
    if st.session_state.matcher:
        st.success("✅ Journal Database")
        stats = st.session_state.matcher.get_database_stats()
        st.write(f"**Journals:** {stats.get('total_journals', 0)}")
        st.write(f"**Index:** {stats.get('faiss_index_type', 'Unknown')}")
    else:
        st.warning("⚠️ Journal Database")
        st.write("Click 'Initialize System' above")


def find_matching_journals(abstract_text):
    """Find and display matching journals for given abstract."""
    if not st.session_state.matcher:
        st.error("❌ System not initialized. Please click 'Initialize System' in the sidebar.")
        return
    
    try:
        st.header("🎯 Journal Matching Results")
        
        # Search parameters
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Number of results", 1, 20, 10)
        with col2:
            min_similarity = st.slider("Min similarity", 0.0, 1.0, 0.0, 0.1)
        
        # Enhanced filtering options
        st.subheader("🔍 Filter Options")
        
        # SJR Quality & Ranking Filters (NEW CSV Features)
        st.markdown("**📊 Journal Quality & Rankings (Scimago)**")
        col1, col2, col3 = st.columns(3)
        with col1:
            sjr_quartiles = st.multiselect(
                "SJR Quartiles", 
                ["Q1", "Q2", "Q3", "Q4"],
                help="Q1 = Top 25%, Q2 = 25-50%, Q3 = 50-75%, Q4 = Bottom 25%"
            )
        with col2:
            max_rank = st.number_input(
                "Max Scimago Rank", 
                min_value=1, max_value=7648, value=7648, step=100,
                help="Only journals ranked within this position (1 = best)"
            )
        with col3:
            min_sjr_score = st.number_input(
                "Min SJR Score", 
                min_value=0.0, value=0.0, step=0.1,
                help="Minimum SJR impact score"
            )
        
        # Open Access Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🔓 Open Access**")
            open_access_only = st.checkbox("Open Access Only")
            doaj_only = st.checkbox("DOAJ Listed Only", help="Only journals listed in Directory of Open Access Journals")
        
        with col2:
            st.markdown("**💰 Publication Costs**")
            no_apc_only = st.checkbox("Free to Publish", help="No Article Processing Charges")
            
            apc_range = st.selectbox(
                "APC Range (USD)",
                ["Any", "Under $500", "$500-$1000", "$1000-$2000", "$2000-$3000", "Over $3000"],
                help="Article Processing Charge limits"
            )
        
        with col3:
            st.markdown("**📈 Impact Metrics**")
            min_citations = st.number_input("Min Citations", min_value=0, value=0, step=1000, 
                                          help="Minimum citation count")
            min_h_index = st.number_input("Min H-Index", min_value=0, value=0, step=10,
                                        help="Minimum h-index")
        
        # Advanced filters in expandable section
        with st.expander("🎛️ Advanced Filters"):
            # Subject filter
            subject_filter = st.text_input("Subject Keywords", 
                                         placeholder="e.g., biology, computer science, medicine",
                                         help="Comma-separated subject areas to match")
            
            # Publisher filter
            publisher_filter = st.text_input("Publisher Keywords",
                                           placeholder="e.g., Elsevier, Springer, PLOS",
                                           help="Comma-separated publisher names to match")
            
            # Language filter
            language_filter = st.multiselect("Languages", 
                                           ["English", "Spanish", "French", "German", "Portuguese", "Chinese"],
                                           help="Journal publication languages")
        
        # Build filters dictionary
        filters = {}
        
        # NEW: SJR Quality & Ranking filters
        if sjr_quartiles:
            filters['sjr_quartiles'] = sjr_quartiles
        if max_rank < 7648:
            filters['max_rank'] = max_rank
        if min_sjr_score > 0:
            filters['min_sjr_score'] = min_sjr_score
        
        # Open access filters
        if open_access_only:
            filters['open_access_only'] = True
        if doaj_only:
            filters['doaj_only'] = True
        if no_apc_only:
            filters['no_apc_only'] = True
            
        # APC range filter
        if apc_range != "Any":
            if apc_range == "Under $500":
                filters['max_apc'] = 500
            elif apc_range == "$500-$1000":
                filters['min_apc'] = 500
                filters['max_apc'] = 1000
            elif apc_range == "$1000-$2000":
                filters['min_apc'] = 1000
                filters['max_apc'] = 2000
            elif apc_range == "$2000-$3000":
                filters['min_apc'] = 2000
                filters['max_apc'] = 3000
            elif apc_range == "Over $3000":
                filters['min_apc'] = 3000
        
        # Citation and h-index filters
        if min_citations > 0:
            filters['min_citations'] = min_citations
        if min_h_index > 0:
            filters['min_h_index'] = min_h_index
            
        # Advanced filters
        if subject_filter.strip():
            filters['subjects'] = [s.strip() for s in subject_filter.split(',')]
        if publisher_filter.strip():
            filters['publishers'] = [p.strip() for p in publisher_filter.split(',')]
        if language_filter:
            filters['languages'] = language_filter
        
        # Display active filters
        if filters:
            st.info(f"🔍 Active filters: {len(filters)} filter(s) applied")
        
        # Perform search
        with st.spinner("Searching for matching journals..."):
            st.write(f"Searching with {len(filters)} filters applied..." if filters else "Searching all journals...")
            
            results = st.session_state.matcher.search_similar_journals(
                query_text=abstract_text,
                top_k=top_k,
                min_similarity=min_similarity,
                filters=filters if filters else None
            )
        
        if results:
            st.success(f"✅ Found {len(results)} matching journals")
            
            # Format results for display
            formatted_results = format_search_results(results)
            
            # Display results as enhanced cards
            for i, result in enumerate(formatted_results, 1):
                with st.container():
                    # Journal header with enhanced info
                    col_title, col_doaj = st.columns([3, 1])
                    with col_title:
                        st.markdown(f"### {i}. {result['journal_name']}")
                    with col_doaj:
                        if result.get('in_doaj'):
                            st.markdown("🏆 **DOAJ Listed**")
                    
                    # Main metrics with CSV-specific data
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    with col1:
                        st.metric("Similarity", f"{result['similarity_score']:.3f}")
                    with col2:
                        # NEW: SJR Rank & Quartile
                        sjr_rank = result.get('scimago_rank')
                        sjr_quartile = result.get('sjr_quartile')
                        if sjr_rank and sjr_quartile:
                            rank_display = f"#{sjr_rank} ({sjr_quartile})"
                            # Add quartile emoji
                            if sjr_quartile == 'Q1':
                                rank_display = f"🥇 {rank_display}"
                            elif sjr_quartile == 'Q2':
                                rank_display = f"🥈 {rank_display}"
                            elif sjr_quartile == 'Q3':
                                rank_display = f"🥉 {rank_display}"
                            else:
                                rank_display = f"📊 {rank_display}"
                        else:
                            rank_display = "N/A"
                        st.metric("SJR Rank", rank_display)
                    with col3:
                        # NEW: SJR Score
                        sjr_score = result.get('sjr_score')
                        st.metric("SJR Score", f"{sjr_score:.2f}" if sjr_score else "N/A")
                    with col4:
                        st.metric("Publisher", result['publisher'][:20] + "..." if len(result['publisher']) > 20 else result['publisher'])
                    with col5:
                        if result['is_open_access']:
                            oa_status = "✅ OA"
                            if result.get('oa_start_year'):
                                oa_status += f" ({result['oa_start_year']})"
                        else:
                            oa_status = "❌ Sub"
                        st.metric("Access", oa_status)
                    with col6:
                        # Enhanced APC display
                        if result.get('apc_display'):
                            apc_display = f"${result['apc_display']}"
                        elif result.get('has_apc') == False:
                            apc_display = "💚 Free"
                        elif result.get('apc_amount') == 0:
                            apc_display = "💚 Free"
                        else:
                            apc_display = "N/A"
                        st.metric("APC", apc_display)
                    
                    # Secondary metrics with CSV data
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        h_index = result.get('h_index', 0)
                        st.metric("H-Index", h_index if h_index > 0 else "N/A")
                    with col2:
                        citations = result.get('cited_by_count', 0)
                        if citations > 0:
                            if citations >= 1000000:
                                cit_display = f"{citations/1000000:.1f}M"
                            elif citations >= 1000:
                                cit_display = f"{citations/1000:.0f}K"
                            else:
                                cit_display = str(citations)
                        else:
                            cit_display = "N/A"
                        st.metric("Citations", cit_display)
                    with col3:
                        works = result.get('works_count', 0)
                        if works > 0:
                            if works >= 1000000:
                                works_display = f"{works/1000000:.1f}M"
                            elif works >= 1000:
                                works_display = f"{works/1000:.0f}K"
                            else:
                                works_display = str(works)
                        else:
                            works_display = "N/A"
                        st.metric("Publications", works_display)
                    with col4:
                        country = result.get('country')
                        st.metric("Country", country if country else "N/A")
                    with col5:
                        # Show region for CSV data
                        region = result.get('region', '')
                        if region:
                            region_display = region[:15] + "..." if len(region) > 15 else region
                        else:
                            region_display = "N/A"
                        st.metric("Region", region_display)
                    
                    # Additional details in expandable section
                    with st.expander(f"📋 More Details - {result['journal_name']}"):
                        detail_col1, detail_col2 = st.columns(2)
                        
                        with detail_col1:
                            # Subject areas
                            if result['subjects']:
                                st.write(f"**Subject Areas:** {', '.join(result['subjects'])}")
                            
                            # ISSN
                            if result.get('issn'):
                                st.write(f"**ISSN:** {result['issn']}")
                            
                            # License information
                            if result.get('license_type'):
                                licenses = ', '.join(result['license_type'])
                                st.write(f"**License Types:** {licenses}")
                        
                        with detail_col2:
                            # Additional DOAJ info
                            if result.get('in_doaj'):
                                st.write("✅ **Listed in DOAJ** (Directory of Open Access Journals)")
                            
                            # APC details
                            if result.get('has_apc') == False:
                                st.write("💚 **No Article Processing Charges**")
                            elif result.get('apc_amount') and result.get('apc_currency'):
                                st.write(f"💰 **APC:** {result['apc_amount']} {result['apc_currency']}")
                    
                    # Action buttons
                    if result['homepage_url']:
                        st.markdown(f"🔗 [Visit Journal Homepage]({result['homepage_url']})")
                    
                    st.divider()  # Separator between results
            
            # Export option
            if st.button("📥 Export Results as CSV"):
                df = pd.DataFrame(formatted_results)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="journal_matches.csv",
                    mime="text/csv"
                )
        else:
            st.warning("⚠️ No matching journals found. Try lowering the similarity threshold.")
    
    except MatchingError as e:
        st.error(f"❌ Matching failed: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.code(traceback.format_exc())


def display_database_info():
    """Display information about the journal database."""
    st.header("📊 Journal Database Information")
    
    try:
        # Load database stats
        journals, embeddings = load_journal_database()
        
        # Enhanced statistics with CSV data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Journals", len(journals))
        with col2:
            st.metric("Embedding Dimension", embeddings.shape[1] if embeddings is not None else 0)
        with col3:
            # Count Q1 journals
            q1_count = sum(1 for j in journals if j.get('sjr_quartile') == 'Q1')
            st.metric("Q1 Journals", q1_count)
        with col4:
            # Count top 100 ranked journals
            top_100_count = sum(1 for j in journals if j.get('scimago_rank', float('inf')) <= 100)
            st.metric("Top 100 Ranked", top_100_count)
        
        # Additional CSV-specific statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            q2_count = sum(1 for j in journals if j.get('sjr_quartile') == 'Q2')
            st.metric("Q2 Journals", q2_count)
        with col2:
            q3_count = sum(1 for j in journals if j.get('sjr_quartile') == 'Q3')
            st.metric("Q3 Journals", q3_count)
        with col3:
            q4_count = sum(1 for j in journals if j.get('sjr_quartile') == 'Q4')
            st.metric("Q4 Journals", q4_count)
        with col4:
            # Count unique countries
            countries = set(j.get('country') for j in journals if j.get('country'))
            st.metric("Countries", len(countries))
        
        # Journal list
        st.subheader("Available Journals")
        
        # Create an enhanced dataframe for display with CSV data
        journal_data = []
        for journal in journals:
            # Format SJR rank and quartile
            sjr_rank = journal.get('scimago_rank')
            sjr_quartile = journal.get('sjr_quartile')
            if sjr_rank and sjr_quartile:
                rank_display = f"#{sjr_rank} ({sjr_quartile})"
            else:
                rank_display = "N/A"
            
            journal_data.append({
                'Name': journal.get('display_name', 'Unknown'),
                'SJR Rank': rank_display,
                'SJR Score': f"{journal.get('sjr_score', 0):.2f}" if journal.get('sjr_score') else "N/A",
                'H-Index': journal.get('h_index', 0) or "N/A",
                'Publisher': journal.get('publisher', 'Unknown'),
                'Country': journal.get('country', 'Unknown'),
                'Articles': f"{journal.get('works_count', 0):,}",
                'Citations': f"{journal.get('cited_by_count', 0):,}",
            })
        
        df = pd.DataFrame(journal_data)
        st.dataframe(df, use_container_width=True)
        
        # Sample journal details
        if journals:
            st.subheader("Sample Journal Detail")
            sample_journal = journals[0]
            
            with st.expander(f"📖 {sample_journal.get('display_name', 'Unknown')}"):
                st.write(f"**Publisher:** {sample_journal.get('publisher', 'Unknown')}")
                st.write(f"**Homepage:** {sample_journal.get('homepage_url', 'N/A')}")
                st.write(f"**Type:** {sample_journal.get('type', 'Unknown')}")
                st.write(f"**Country:** {sample_journal.get('country_code', 'N/A')}")
                
                if sample_journal.get('semantic_fingerprint'):
                    st.write("**Semantic Fingerprint:**")
                    st.text(sample_journal['semantic_fingerprint'][:300] + "...")
    
    except Exception as e:
        st.error(f"❌ Failed to load database info: {e}")


if __name__ == "__main__":
    main()