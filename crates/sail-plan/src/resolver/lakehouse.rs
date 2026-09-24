use sail_catalog::error::CatalogError;
use sail_catalog::lakehouse::{
    BeginTableAccessRequest, ResolveLakehouseTableRequest, TableAccessPurpose,
};
use sail_catalog::manager::CatalogManager;
use sail_common_datafusion::catalog::{
    LakehouseExecutionContext, LakehouseFormat, LakehouseOperation,
};
use sail_common_datafusion::extension::SessionExtensionAccessor;

use crate::error::PlanResult;
use crate::resolver::PlanResolver;

impl PlanResolver<'_> {
    pub(super) async fn resolve_lakehouse_table_context(
        &self,
        table: &[String],
        operation: LakehouseOperation,
        requested_format: Option<&str>,
        options: Vec<(String, String)>,
    ) -> PlanResult<LakehouseExecutionContext> {
        self.resolve_lakehouse_table_context_impl(
            table,
            operation,
            requested_format,
            options,
            false,
        )
        .await
    }

    /// Like [`Self::resolve_lakehouse_table_context`], for a table read by a query that
    /// writes nothing, so the catalog table cache may serve it. Never use this for a write.
    pub(super) async fn resolve_lakehouse_table_context_for_read(
        &self,
        table: &[String],
        requested_format: Option<&str>,
        options: Vec<(String, String)>,
    ) -> PlanResult<LakehouseExecutionContext> {
        self.resolve_lakehouse_table_context_impl(
            table,
            LakehouseOperation::Read,
            requested_format,
            options,
            true,
        )
        .await
    }

    async fn resolve_lakehouse_table_context_impl(
        &self,
        table: &[String],
        operation: LakehouseOperation,
        requested_format: Option<&str>,
        options: Vec<(String, String)>,
        read_only: bool,
    ) -> PlanResult<LakehouseExecutionContext> {
        let manager = self.ctx.extension::<CatalogManager>()?;
        let request = ResolveLakehouseTableRequest {
            catalog_table: table.to_vec(),
            operation,
            requested_format: requested_format.map(LakehouseFormat::from_format_name),
            options,
        };
        let resolved = if read_only {
            manager
                .resolve_lakehouse_table_for_read(table, request)
                .await?
        } else {
            manager.resolve_lakehouse_table(table, request).await?
        };
        let execution = resolved.execution;
        let request = BeginTableAccessRequest {
            context: execution.clone(),
            purpose: table_access_purpose(operation),
        };
        let session = if read_only {
            manager.begin_table_access_for_read(table, request).await
        } else {
            manager.begin_table_access(table, request).await
        };
        match session {
            Ok(session) => Ok(session.context),
            Err(CatalogError::NotSupported(_) | CatalogError::UnsupportedCapability(_)) => {
                Ok(execution)
            }
            Err(error) => Err(error.into()),
        }
    }
}

fn table_access_purpose(operation: LakehouseOperation) -> TableAccessPurpose {
    match operation {
        LakehouseOperation::Read => TableAccessPurpose::DataRead,
        LakehouseOperation::Write
        | LakehouseOperation::WritePrecondition
        | LakehouseOperation::Create
        | LakehouseOperation::Register => TableAccessPurpose::DataWrite,
        LakehouseOperation::Alter | LakehouseOperation::Maintenance => {
            TableAccessPurpose::MetadataRead
        }
    }
}
