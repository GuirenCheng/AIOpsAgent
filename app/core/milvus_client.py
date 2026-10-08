"""Milvus 客户端工厂模块"""
from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
    MilvusException
)

from app.config import config


class MilvusClientManager:
    '''Milvus客户端管理器'''

    # 常量定义
    COLLECTION_NAME: str = 'biz'
    VECTOR_DIM: int = config.milvus_vector_dim  # 统一使用配置中的维度（唯一来源）
    ID_MAX_LENGTH: int = 100
    CONTENT_MAX_LENGTH:int = 8000
    DEFAULT_SHARD_NUMBER:int = 2

    def __init__(self)->None:
        '''初始化Milvus客户端管理器'''
        self._client : MilvusClient | None = None
        self._collection : Collection | None = None
        # 唯一连接别名：由 MilvusClient(uri=...) 创建，供 Collection/utility 复用。
        # 所有对 biz collection 的读写都走这一条连接，避免多条并行连接。
        self._alias : str | None = None

    @property
    def connection_uri(self) -> str:
        '''Milvus 服务 URI（唯一连接地址，供其它模块复用）'''
        return f'http://{config.milvus_host}:{config.milvus_port}'

    @property
    def alias(self) -> str | None:
        '''当前使用的唯一连接别名（未连接时为 None）'''
        return self._alias

    def connect(self):
        """
        连接到 Milvus 服务器并初始化 collection

        Returns:
            MilvusClient: Milvus 客户端实例

        Raises:
            RuntimeError: 连接或初始化失败时抛出
        """
        try:
            logger.info(f'正在连接到Milvus:{config.milvus_host}:{config.milvus_port}')

            # 创建唯一客户端连接。
            # 不再额外调用 connections.connect(alias='default', ...)，
            # 而是由 MilvusClient(uri=...) 建立唯一一条连接，
            # 后续 Collection / utility 全部通过 self._alias 复用这条连接。
            self._client = MilvusClient(
                uri=self.connection_uri,
                timeout=config.milvus_timeout / 1000,
            )
            self._alias = self._client._using

            logger.info(f'成功连接到Milvus（连接别名：{self._alias}）')

            # 检查并创建Collection
            if not self._collection_exists():
                logger.info(f"collection {self.COLLECTION_NAME}'不存在，正在创建...")
                self._create_collection()
                logger.info(f"成功创建 collection '{self.COLLECTION_NAME}'")
            else:
                logger.info(f"collection '{self.COLLECTION_NAME}'已存在")
                self._collection = Collection(self.COLLECTION_NAME, using=self._alias)


                # 检查向量维度是否匹配
                schema = self._collection.schema
                vector_field = None
                existing_dim =None
                for field in schema.fields:
                    if field.name == 'vector':
                        vector_field = field
                        break

                if vector_field and hasattr(vector_field,'params') and 'dim' in vector_field.params:
                    existing_dim = vector_field.params['dim']
                    if existing_dim != self.VECTOR_DIM:
                        # 数据安全保护：维度不匹配意味着 embedding 模型/维度配置被改动。
                        # 若直接 drop + 重建，会静默丢失 collection 中的全部旧数据。
                        # 只有显式开启 MILVUS_ALLOW_DIMENSION_MIGRATION 时才执行破坏性迁移。
                        if not config.milvus_allow_dimension_migration:
                            self.close()
                            raise RuntimeError(
                                f"向量维度不匹配：collection '{self.COLLECTION_NAME}' 现有维度 "
                                f"{existing_dim}，而配置维度为 {self.VECTOR_DIM}。"
                                f"为避免静默丢失数据，已拒绝启动。"
                                f"请先手动迁移数据，或（确认数据可丢弃后）设置环境变量 "
                                f"MILVUS_ALLOW_DIMENSION_MIGRATION=true 以允许自动删除并重建。"
                            )
                        logger.warning(
                            f"检测到向量维度不匹配且已开启迁移开关！当前collection维度：{existing_dim},"
                            f"配置维度：{self.VECTOR_DIM}，即将删除并重建。"
                        )
                        logger.info(f"正在删除旧collection '{self.COLLECTION_NAME}'...")
                        _ = utility.drop_collection(self.COLLECTION_NAME, using=self._alias)
                        logger.info(f"正在重新创建collection '{self.COLLECTION_NAME}'...")
                        self._create_collection()
                        logger.info(f"成功创建collection，维度：{self.VECTOR_DIM}")
                    else:
                        logger.info(f"向量维度匹配：{self.VECTOR_DIM}")

            # 加载collection
            self._load_collection()

            return self._client

        except MilvusException as e:
            logger.error(f"Milvus操作失败：{e}")
            self.close()
            raise RuntimeError(f"Milvus操作失败：{e}") from e
        except ConnectionError as e:
            logger.error(f"连接Milvus失败：{e}")
            self.close()
            raise RuntimeError(f"连接Milvus失败：{e}") from e
        except Exception as e:
            logger.error(f"连接Milvus失败：{e}")
            self.close()
            raise RuntimeError(f"连接Milvus失败：{e}") from e

    def _collection_exists(self) -> bool:
        '''检查collection是否存在'''
        # pymilvus的类型注解可能不准确，实际返回bool
        result = utility.has_collection(self.COLLECTION_NAME, using=self._alias)
        return bool(result) # type: ignore[arg-type]

    def _create_collection(self) :
        '''创建biz collection'''
        # 定义字段
        fields = [
            FieldSchema(
                name="id",
                dtype = DataType.VARCHAR,
                max_length = self.ID_MAX_LENGTH,
                is_primary = True,
            ),
            FieldSchema(
                name = 'vector',
                dtype = DataType.FLOAT_VECTOR,
                dim = self.VECTOR_DIM,
            ),
            FieldSchema(
                name = 'content',
                dtype = DataType.VARCHAR,
                max_length = self.CONTENT_MAX_LENGTH,
            ),
            FieldSchema(
                name = 'metadata',
                dtype = DataType.JSON
            ),
        ]

        # 创建schema
        schema =CollectionSchema(
            fields = fields,
            description = 'Business knowledge collection',
            enable_dynamic_fields = True,
        )

        # 分片配置放到字典
        collection_params = {
            "num_shards": self.DEFAULT_SHARD_NUMBER
        }

        # 创建collection（复用唯一连接别名）
        self._collection = Collection(
            name=self.COLLECTION_NAME,
            schema=schema,
            using=self._alias,
            collection_params=collection_params
        )

        # 创建索引
        self._create_index()

    def _create_index(self)->None:
        '''为vector字段创建索引'''
        if self._collection is None:
            raise RuntimeError("Collection 未初始化")

        index_params = {
            'metric_type': "L2",
            'index_type': 'IVF_FLAT',
            'params':{'nlist':128}
        }

        _ = self._collection.create_index(
            field_name = 'vector',
            index_params = index_params,
        )
        logger.info('成功为vector字段创建索引')

    def _load_collection(self)->None:
        '''加载collection 到内存'''
        if self._collection is None:
            self._collection = Collection(self.COLLECTION_NAME, using=self._alias)

        # 检查collection是否已加载（兼容多版本）
        try:
            # 方法1：尝试使用utility.load_state(新版本）
            load_state = utility.load_state(self.COLLECTION_NAME, using=self._alias)
            # load_state 返回字符串或枚举，如“Loaded”或“NotLoad”
            state_name = getattr(load_state,'name',str(load_state))
            if state_name != "Loaded":
                self._collection.load()
                logger.info(f"成功加载collection '{self.COLLECTION_NAME}'")
            else:
                logger.info(f"Collection '{self.COLLECTION_NAME}'已加载")
        except AttributeError:
            # 方法2，直接尝试加载，捕获‘already loaded’异常
            try:
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            except MilvusException as e:
                error_msg = str(e).lower()
                if 'already loaded' in error_msg or 'loaded' in error_msg:
                    logger.info(f"Colllection '{self.COLLECTION_NAME}' 已加载")
                else:
                    raise
        except Exception as e:
            logger.error(f"加载collection失败：{e}")
            raise

    def get_collection(self)->Collection:
        '''
        获取collection实例

        :return:
            Collection：collection实例

        Raises:
        RuntioneError:collection 未初始化时抛出
        '''
        if self._collection is None:
            raise RuntimeError("Collection 未初始化，请先调用connect()")
        return self._collection

    def health_check(self)->bool:
        '''
        健康检查
        :return:
            bool：True代表健康，False表示异常
        '''
        try:
            if self._client is None or self._alias is None:
                return False

            # 校验唯一连接别名仍在连接池中
            return bool(connections.has_connection(self._alias))

        except (MilvusException,ConnectionError) as e:
            logger.error(f"Milvus健康检查失败：{e}")
            return False
        except Exception as e:
            logger.error(f"Milvus 健康检查失败：{e}")
            return False

    def close(self) ->None:
        errors = []

        try:
            if self._collection is not None:
                self._collection.release()
                self._collection = None

        except Exception as e:
            errors.append(f"释放 collection 失败：{e}")

        try:
            if self._alias and connections.has_connection(self._alias):
                connections.disconnect(self._alias)
        except Exception as e:
            errors.append(f"断开连接失败：{e}")

        self._client = None
        self._alias = None

        if errors:
            error_msg = '; '.join(errors)
            logger.error(f"关闭Milvus 连接时出现错误：{error_msg}")
        else:
            logger.info('已关闭Milvus')

    def __enter__(self)->"MilvusClientManager":
        '''上下文管理器入口'''
        _ = self.connect()
        return self

    def __exit__(
        self,
        exc_type:type[BaseException] | None,
        exc_val:BaseException | None,
        exc_tb:object
    )->None:
        '''上下文管理器退出'''
        self.close()

# 全局单例
milvus_manager = MilvusClientManager()