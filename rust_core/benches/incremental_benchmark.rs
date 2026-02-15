use criterion::{criterion_group, criterion_main, Criterion};
use std::fs;
use std::path::Path;
use semantic_engine::incremental_parser::{IncrementalParser, TextEdit};
use semantic_engine::parser::SupportedLanguage;

fn benchmark_incremental_parsing(c: &mut Criterion) {
    let file_path = Path::new("../test_repo/packages/react-devtools-shared/src/backend/fiber/renderer.js");
    let source_code = fs::read_to_string(file_path).expect("Could not read file");

    let mut group = c.benchmark_group("Incremental Parsing");

    // Benchmark 1: Initial full parse
    group.bench_function("Initial Full Parse", |b| {
        b.iter(|| {
            let mut parser = IncrementalParser::new();
            parser.parse(&source_code, SupportedLanguage::JavaScript).unwrap();
        })
    });

    let mut initial_parser = IncrementalParser::new();
    initial_parser.parse(&source_code, SupportedLanguage::JavaScript);

    // Benchmark 2: Small edit in the middle
    let edit = TextEdit {
        start_line: 1000,
        start_col: 0,
        end_line: 1000,
        end_col: 0,
        old_text: "".to_string(),
        new_text: "// BENCHMARK INSERTION POINT\n".to_string(),
    };
    let mut new_source_code = source_code.clone();
    let byte_offset = IncrementalParser::position_to_byte(&source_code, edit.start_line, edit.start_col);
    new_source_code.insert_str(
        byte_offset,
        &edit.new_text,
    );

    let path = Path::new("test.js");
    let mut parser_for_update = IncrementalParser::new();
    parser_for_update.parse(&source_code, SupportedLanguage::JavaScript);
    let _ = parser_for_update.update_file(path, source_code.clone(), &TextEdit{
        start_line: 0,
        start_col: 0,
        end_line: 0,
        end_col: 0,
        old_text: "".to_string(),
        new_text: "".to_string(),
    });


    group.bench_function("Incremental Edit (Small)", |b| {
        b.iter(|| {
            let mut p = parser_for_update.clone();
            p.update_file(path, new_source_code.clone(), &edit).unwrap();
        })
    });

    // Benchmark 3: No-op edit
    let noop_edit = TextEdit {
        start_line: 1,
        start_col: 0,
        end_line: 1,
        end_col: 0,
        old_text: "".to_string(),
        new_text: "".to_string(),
    };
    group.bench_function("Incremental Edit (No-op)", |b| {
        b.iter(|| {
            let mut p = parser_for_update.clone();
            p.update_file(path, source_code.clone(), &noop_edit).unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_incremental_parsing);
criterion_main!(benches);
