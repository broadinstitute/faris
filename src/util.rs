pub(crate) fn format_date_time(dt: &time::OffsetDateTime) -> String {
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}", dt.year(), dt.month() as u8, dt.day(), dt.hour(),
        dt.minute(), dt.second()
    )
}
